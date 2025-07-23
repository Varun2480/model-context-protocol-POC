import os
import sys
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import google.generativeai as genai
# from google import genai
import json # For handling tool_args if they are JSON strings
from typing import Optional, List, Dict, Any

from mcp import ClientSession, StdioServerParameters
# Alias 'types' from mcp.common to avoid conflict with Python's built-in 'types' module
from mcp import types as mcp_common_types

from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


# --- Configuration (same as previous examples) ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.gemini_model: Optional[genai.GenerativeModel] = None
        self.gemini_chat_session: Optional[genai.ChatSession] = None
    # methods will go here

    # This is the cleanup method that was apparently missing or misplaced
    async def cleanup(self):
        """Perform graceful shutdown of all resources."""
        print("Performing client cleanup...")
        if self.gemini_chat_session:
            # You might want to save chat history here if needed
            self.gemini_chat_session = None
        if self.gemini_model:
            self.gemini_model = None

        # Disconnect from MCP server (this will also close the stdio transport)
        await self.disconnect_from_mcp_server() # This method already calls self.exit_stack.aclose()

        print("Client cleanup complete.")


    async def connect_to_mcp_server(self, server_command: str, *server_args: str):
        """
        Connects to the MCP server using stdio transport.
        You might need to adjust this based on your MCP server's transport.
        """
        print(f"Attempting to connect to MCP server with command: {server_command} {' '.join(server_args)}")
        try:
            # Use AsyncExitStack to ensure cleanup of the session
            # This sets up the transport for the MCP client
            transport = await self.exit_stack.enter_async_context(
                stdio_client(StdioServerParameters(command=server_command, args=list(server_args)))
            )
            read, write, *_ = transport # stdio_client might return more than read/write

            # Initialize the ClientSession
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            # Perform initial handshake with the server
            await self.session.initialize()
            print("Successfully connected to MCP server.")
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            self.session = None # Ensure session is None on failure
            raise # Re-raise to indicate connection failure

    
    async def disconnect_from_mcp_server(self):
        """Disconnection cleanup."""
        print("Disconnecting from MCP server...")
        await self.exit_stack.aclose()
        self.session = None
        print("Disconnected from MCP server.")


    async def _get_gemini_function_declarations(self) -> List[genai.types.FunctionDeclaration]:
        """
        Fetches tools from the MCP session and converts them into Gemini's FunctionDeclaration format.
        """
        if not self.session:
            raise RuntimeError("MCP session is not connected. Call connect_to_mcp_server first.")

        print("[MCPClient] Fetching tools from MCP server...")
        response = await self.session.list_tools()
        
        gemini_tools = []
        # MCP's list_tools() should return a list of MCPTool objects or similar structure
        # We need to adapt that structure to genai.types.FunctionDeclaration
        for tool_obj in response.tools: # Assuming `response.tools` is the list
            try:
                # MCP's inputSchema should be an OpenAPI Schema Object compatible dictionary
                input_schema_dict = tool_obj.inputSchema # Assuming inputSchema is directly available
                
                # Basic validation for common schema properties expected by Gemini
                if not isinstance(input_schema_dict, dict):
                    input_schema_dict = {} # Default to empty if not a dict

                # Create a mutable copy to modify
                cleaned_input_schema = input_schema_dict.copy()

                # List of unsupported/unnecessary top-level schema keywords for Gemini's FunctionDeclaration.parameters
                unsupported_keywords = ['additionalProperties', '$schema']
                
                for keyword in unsupported_keywords:
                    if keyword in cleaned_input_schema:
                        # print(f"  - Warning: Removing '{keyword}' from schema for tool '{tool_obj.name}'.")
                        del cleaned_input_schema[keyword]
                
                if "type" not in cleaned_input_schema:
                    cleaned_input_schema["type"] = "object" # Default to object type
                if "properties" not in cleaned_input_schema:
                    cleaned_input_schema["properties"] = {}
                if "required" not in cleaned_input_schema:
                    cleaned_input_schema["required"] = []

                gemini_tools.append(
                    genai.types.FunctionDeclaration(
                        name=tool_obj.name,
                        description=tool_obj.description,
                        parameters=cleaned_input_schema # Pass the dictionary directly
                    )
                )
                # print(f"  - Converted MCP tool '{tool_obj.name}' to Gemini FunctionDeclaration.")
            except AttributeError as e:
                print(f"  - Warning: MCP tool object missing expected attribute (e.g., .name, .description, .inputSchema): {e}. Tool: {tool_obj}")
            except Exception as e:
                print(f"  - Error converting MCP tool {tool_obj.name if hasattr(tool_obj, 'name') else 'N/A'}: {e}")
        
        print(f"[MCPClient] Converted {len(gemini_tools)} tools for Gemini.")
        # Store the dynamically loaded tools so that other methods (like main) can access them.
        self.dynamic_tools_loaded = gemini_tools # <--- ADD THIS LINE
        return gemini_tools


    async def process_query(self, query: str) -> str:        
        """Process a query using Gemini 1.5 Flash and dynamically available MCP tools."""
        if not self.session:
            raise RuntimeError("MCP session is not connected. Please connect to the MCP server first.")

        # Dynamically get tools from the MCP session for this query.
        # This will refresh the tools if they change on the MCP server side
        # And importantly, it will update self.dynamic_tools_loaded
        available_gemini_tools = await self._get_gemini_function_declarations()

        # Check if the model needs to be initialized or re-initialized with new tools
        model_needs_reinitialization = False
        if not self.gemini_model:
            model_needs_reinitialization = True
            print(f"Initializing Gemini model with {len(available_gemini_tools)} dynamic tools for the first time.")
        else:
            # Safely check if self.gemini_model.tools exists and compare
            # Note: The 'tools' attribute on GenerativeModel is a collection of FunctionDeclaration objects.
            # Comparing by names is a practical way to detect changes.
            current_tool_names = {t.name for t in (self.gemini_model.tools.tools if hasattr(self.gemini_model, 'tools') and self.gemini_model.tools else [])}
            new_tool_names = {t.name for t in available_gemini_tools}

            if current_tool_names != new_tool_names:
                model_needs_reinitialization = True
                print("[CLIENT] Detected tool changes. Re-initializing Gemini model...")
            # else: no change, keep current model and chat session

        if model_needs_reinitialization:
            try:
                self.gemini_model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    tools=available_gemini_tools # Pass the actively fetched tools
                )
                # Re-start a chat session associated with the new model
                # Note: Re-initializing model means chat history is lost unless explicitly managed/transferred.
                # For long-running chats, you might want to save and re-load history.
                self.gemini_chat_session = self.gemini_model.start_chat(history=[])
                print(f"Gemini model {'re-initialized' if not self.gemini_model else 'initialized'} with {len(available_gemini_tools)} dynamic tools.")
            except Exception as e:
                print(f"Error during Gemini model initialization/re-initialization: {e}")
                # Fallback: if model initialization fails, ensure a basic model (without tools) is available
                # to at least allow text-only interaction or prevent complete crash.
                if not self.gemini_model or not hasattr(self.gemini_model, 'tools') or not self.gemini_model.tools:
                    print("Falling back to a Gemini model without tools due to initialization error.")
                    self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                    self.gemini_chat_session = self.gemini_model.start_chat(history=[])

        # If after all attempts, gemini_model or gemini_chat_session is still not set,
        # it means a critical error occurred and we cannot proceed.
        if not self.gemini_model or not self.gemini_chat_session:
            raise RuntimeError("Gemini model or chat session could not be initialized. Cannot process query.")


        print(f"\n--- Processing Query: '{query}' ---")

        try:
            print("AI: ", end="", flush=True)
            response_stream = await self.gemini_chat_session.send_message_async(query, stream=True)

            full_response_text = ""
            tool_calls_made = []

            async for chunk in response_stream:
                # Iterate over individual parts within the chunk
                for part in chunk.candidates[0].content.parts: # Accessing parts directly
                    if hasattr(part, 'text') and part.text:
                        print(part.text, end="", flush=True)
                        full_response_text += part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        tool_calls_made.append(part.function_call)
                    # You might also want to handle other part types if they occur
                    # e.g., elif hasattr(part, 'function_response'):
                    #     print(f"[DEBUG] Received function_response part: {part.function_response}")
                    else:
                        # This 'else' block will catch any part that is neither text nor function_call
                        # This is where the error "Could not convert part.function_call to text" might be implicitly happening
                        # if the internal `chunk.text` property attempts to represent unsupported part types as text.
                        print(f"[DEBUG] Received unhandled part type in chunk: {type(part)} - {part}")


            print() # Newline after streaming

            if tool_calls_made:
                print(f"[MODEL] Detected tool calls: {[tc.name for tc in tool_calls_made]}")
                tool_results_to_send = []

                for tool_call in tool_calls_made:
                    tool_name = tool_call.name
                    tool_args = dict(tool_call.args) # Convert protobuf map to Python dict

                    print(f"[CLIENT] Requesting MCP session to call tool: {tool_name} with args: {tool_args}")

                    try:
                        tool_result_obj = await self.session.call_tool(tool_name, tool_args)
                        
                        if isinstance(tool_result_obj.content, mcp_common_types.TextContent):
                            result_content = tool_result_obj.content.text
                        elif isinstance(tool_result_obj.content, mcp_common_types.JsonContent):
                            result_content = json.dumps(tool_result_obj.content.value)
                        else:
                            result_content = str(tool_result_obj.content) # Fallback

                        tool_results_to_send.append(
                            # genai.types.Part.from_function_response(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"result": result_content} 
                            )
                        )
                        print(f"[TOOL RESULT] {tool_name} returned: {result_content}")

                    except Exception as tool_error:
                        error_message = f"Error executing tool {tool_name} via MCP session: {tool_error}"
                        print(f"[TOOL ERROR] {error_message}")
                        tool_results_to_send.append(
                            # genai.types.Part.from_function_response(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"error": error_message}
                            )
                        )

                print("[MODEL] Sending tool results back to the model for final response...")
                follow_up_response_stream = await self.gemini_chat_session.send_message_async(
                    contents=tool_results_to_send, # List of Parts
                    stream=True
                )

                final_text_after_tool = []
                print("AI (after tool): ", end="", flush=True)
                async for chunk_after_tool in follow_up_response_stream:
                    # Again, iterate through parts for safety
                    for part_after_tool in chunk_after_tool.candidates[0].content.parts:
                        if hasattr(part_after_tool, 'text') and part_after_tool.text:
                            print(part_after_tool.text, end="", flush=True)
                            final_text_after_tool.append(part_after_tool.text)
                        # If a tool call comes back here, you might need another loop
                        # For simplicity, assuming final response is text or error
                        elif hasattr(part_after_tool, 'function_call') and part_after_tool.function_call:
                             print(f"[DEBUG] Model requested another tool call after tool execution: {part_after_tool.function_call}")
                             # You would typically re-enter the tool execution loop here, or handle it as an error
                             final_text_after_tool.append(f"[ERROR] Model requested another tool call: {part_after_tool.function_call}")


                print() # Newline after streaming

                return "".join(final_text_after_tool)

            else:
                # No tool call, return the initial AI response
                return full_response_text

        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            return f"Error: Could not process query. {e}"

        
    async def chat_loop(self):
        """Run an interactive chat loop with MCP server and Gemini."""
        print("\n" + "="*50)
        print("MCP Gemini Client Started!")
        print("Type your queries or 'quit' to exit.")
        print("="*50 + "\n")

        while True:
            try:
                query = input("You: ").strip() # Changed prompt from "Query: " to "You: "

                if query.lower() == 'quit':
                    break

                # process_query already handles printing AI response and tool call logs
                await self.process_query(query)

            except Exception as e:
                print(f"\nError in chat loop: {str(e)}")
                # Potentially try to reconnect or offer options here


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script_or_command>")
        print("  Example: python client.py \"uvx python -m mcp.server.example\"")
        print("  Or if your server is just a script: python client.py your_server_script.py")
        sys.exit(1)
    
    client = MCPClient()
    
    # sys.argv[1] can be the entire command string (e.g., "uvx python -m mcp.server.example")
    # or just the path to a script. We split it correctly.
    server_command_parts = sys.argv[1].split() 
    server_executable = server_command_parts[0]
    server_args = server_command_parts[1:]

    try:
        # Connect to the MCP server using the provided command and arguments
        await client.connect_to_mcp_server(server_executable, *server_args)
    except Exception as e:
        print(f"Failed to connect to MCP server with command '{sys.argv[1]}': {e}")
        print("Please ensure your MCP server is running, the command is correct, and necessary dependencies are installed.")
        sys.exit(1) # Exit if connection fails

    # Initialize Gemini model with tools right after MCP connection, before chat starts
    # This ensures the model is ready when the first query comes.
    try:
        # Fetch tools once to initialize the model
        initial_tools = await client._get_gemini_function_declarations() 
        
        client.gemini_model = genai.GenerativeModel(
            "gemini-1.5-flash",
            tools=initial_tools 
        )
        client.gemini_chat_session = client.gemini_model.start_chat(history=[])
        print("Gemini model initialized with dynamic tools from MCP server.")
    except Exception as e:
        print(f"Error initializing Gemini model with MCP tools: {e}")
        # You might want to continue without tools or exit here
        print("Continuing without full AI capabilities due to initialization error.")
        # Ensure a basic model is still available if you want to proceed without tools
        if not client.gemini_model:
            client.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            client.gemini_chat_session = client.gemini_model.start_chat(history=[])

    try:
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())