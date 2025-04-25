import asyncio
import os
import sys
import json # Needed for parsing potential JSON results if not already dicts
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, ToolDefinition
from mcp.client.stdio import stdio_client

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool # Optional: If you had static tools
from langchain_core.utils.function_calling import convert_to_openai_tool # Helper

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# --- Environment Variable Checks ---
google_api_key_loaded = os.getenv("GOOGLE_API_KEY") is not None
print(f"Google API Key Loaded: {google_api_key_loaded}")
if not google_api_key_loaded:
    print("Error: GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
    sys.exit(1)
# Optional: Keep or remove Anthropic check based on your needs
# print(f"Anthropic API Key Loaded: {os.getenv('ANTHROPIC_API_KEY') is not None}")
# --- End Environment Variable Checks ---


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Initialize LangChain LLM (Gemini Flash)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            # temperature=0.7, # Optional: Adjust creativity
            convert_system_message_to_human=True # Good practice for some models
        )
        self.mcp_tools: List[ToolDefinition] = [] # Store MCP tool definitions
        self.langchain_tools: List[Dict[str, Any]] = [] # Store Langchain/OpenAI formatted tools

    def _mcp_tool_to_langchain_tool(self, mcp_tool: ToolDefinition) -> Dict[str, Any]:
        """Converts an MCP ToolDefinition to LangChain/OpenAI tool format."""
        # Ensure inputSchema is treated as the parameters definition
        parameters = mcp_tool.inputSchema if mcp_tool.inputSchema else {"type": "object", "properties": {}}

        # Basic validation/correction for schema format if needed
        if not isinstance(parameters, dict):
             print(f"Warning: Tool '{mcp_tool.name}' has non-dict inputSchema: {parameters}. Attempting to parse as JSON.")
             try:
                 parameters = json.loads(str(parameters)) # Attempt parsing if it's a string representation
             except json.JSONDecodeError:
                 print(f"Error: Could not parse inputSchema for tool '{mcp_tool.name}'. Using empty schema.")
                 parameters = {"type": "object", "properties": {}}
        if "type" not in parameters:
             parameters["type"] = "object" # Ensure base type is present
        if "properties" not in parameters:
             parameters["properties"] = {} # Ensure properties key exists

        return convert_to_openai_tool({
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": parameters,
        })


    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server and fetch tools."""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            # Pass Google API Key to server environment if needed by tools
            env={"GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")}
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools from MCP server
        response = await self.session.list_tools()
        self.mcp_tools = response.tools
        self.langchain_tools = [self._mcp_tool_to_langchain_tool(tool) for tool in self.mcp_tools]

        print("\nConnected to server with tools:", [tool.name for tool in self.mcp_tools])
        # Optional: Print the converted LangChain tool format for debugging
        # print("\nLangChain tool format:", json.dumps(self.langchain_tools, indent=2))


    async def process_query(self, query: str) -> str:
        """Process a query using Gemini Flash and available MCP tools via LangChain."""
        if not self.session:
            return "Error: Not connected to MCP server."
        if not self.langchain_tools:
            print("Warning: No tools available from the MCP server.")
            # Fallback to basic LLM call if no tools
            llm_without_tools = self.llm
            response = await llm_without_tools.ainvoke([HumanMessage(content=query)])
            return response.content

        # Bind the dynamically fetched tools to the LLM for this call
        llm_with_tools = self.llm.bind_tools(self.langchain_tools)

        # Start conversation history
        messages: List[Any] = [HumanMessage(content=query)]

        # Loop to handle potential sequences of tool calls
        while True:
            print(f"\n> Invoking LLM with {len(messages)} messages...")
            # Use ainvoke for async
            ai_response: AIMessage = await llm_with_tools.ainvoke(messages)
            messages.append(ai_response) # Add AI response to history

            if not ai_response.tool_calls:
                # No tool calls requested, this is the final answer for this turn
                print("> LLM response received (no tool calls).")
                return ai_response.content # Return the text content

            # --- Tool Call Handling ---
            print(f"> LLM requested {len(ai_response.tool_calls)} tool call(s): {[tc['name'] for tc in ai_response.tool_calls]}")
            tool_messages_for_next_turn: List[ToolMessage] = []

            # Execute all tool calls requested in this turn
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"] # Important for LangChain

                print(f"  - Calling MCP tool: {tool_name} with args: {tool_args}")
                try:
                    # Ensure args are a dictionary (LangChain usually provides this)
                    if not isinstance(tool_args, dict):
                         print(f"Warning: Tool args for {tool_name} are not a dict: {tool_args}. Attempting to use as is.")
                         # You might need more robust parsing here depending on Gemini's output format
                         # For example, if it's a JSON string: tool_args = json.loads(tool_args)

                    mcp_result = await self.session.call_tool(tool_name, tool_args)
                    result_content = mcp_result.content

                    # Ensure result content is a string for ToolMessage
                    if not isinstance(result_content, str):
                        print(f"Warning: Tool '{tool_name}' result content is not a string: {type(result_content)}. Converting to JSON string.")
                        try:
                            result_content = json.dumps(result_content)
                        except TypeError:
                             print(f"Error: Could not serialize tool '{tool_name}' result to JSON. Using string representation.")
                             result_content = str(result_content)


                    print(f"  - MCP tool '{tool_name}' result: {result_content[:100]}...") # Log snippet
                    tool_messages_for_next_turn.append(
                        ToolMessage(content=result_content, tool_call_id=tool_call_id)
                    )
                except Exception as e:
                    print(f"  - Error calling MCP tool '{tool_name}': {e}")
                    # Provide an error message back to the LLM
                    tool_messages_for_next_turn.append(
                        ToolMessage(
                            content=f"Error executing tool {tool_name}: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                    )

            # Add all tool results to the message history for the next LLM call
            messages.extend(tool_messages_for_next_turn)
            # The loop continues, invoking the LLM again with the tool results included


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client (LangChain + Gemini) Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                if not query:
                    continue

                response = await self.process_query(query)
                print("\nGemini: " + response)

            except Exception as e:
                print(f"\nAn error occurred in the chat loop: {str(e)}")
                # Optional: Add more detailed error logging or traceback
                # import traceback
                # traceback.print_exc()

    async def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up resources...")
        await self.exit_stack.aclose()
        print("Cleanup complete.")

async def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_server_script.py|.js>")
        sys.exit(1)

    server_script = sys.argv[1]
    if not os.path.exists(server_script):
        print(f"Error: Server script not found at '{server_script}'")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(server_script)
        await client.chat_loop()
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         # Optional: Add more detailed error logging or traceback
         # import traceback
         # traceback.print_exc()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        # Handle KeyboardInterrupt gracefully
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
