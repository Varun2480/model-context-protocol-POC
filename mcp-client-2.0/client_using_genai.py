import os
import sys
import asyncio
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from google import genai
from google.genai import types  # Replace with correct import if needed
# If Tool is not in `types`, e.g., if it's under a different path, adjust the import:
# from google.genai.content import Tool

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters, types as mcp_common_types
from mcp.client.stdio import stdio_client

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# If your "Tool" class is not in the above imports, add this:
# from google.genai.content import Tool
# Or try: dir(types) in a Python shell to see the correct path

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.dynamic_tools_loaded: List[types.FunctionDeclaration] = []
        self.genai_client: Optional[genai.Client] = None
        try:
            self.genai_client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Error initializing Google GenAI Client: {e}")
            raise

    async def cleanup(self):
        print("Performing client cleanup...")
        await self.exit_stack.aclose()
        self.session = None
        print("Client cleanup complete.")

    async def connect_to_mcp_server(self, server_command: str, *server_args: str):
        print(f"Attempting to connect to MCP server with command: {server_command} {' '.join(server_args)}")
        try:
            transport = await self.exit_stack.enter_async_context(
                stdio_client(StdioServerParameters(command=server_command, args=list(server_args)))
            )
            read, write, *_ = transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            print("Successfully connected to MCP server.")
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            self.session = None
            raise

    async def disconnect_from_mcp_server(self):
        print("Disconnecting from MCP server...")
        await self.exit_stack.aclose()
        self.session = None
        print("Disconnected from MCP server.")

    async def _get_gemini_function_declarations(self) -> List[types.FunctionDeclaration]:
        if not self.session:
            raise RuntimeError("MCP session is not connected. Call connect_to_mcp_server first.")

        print("[MCPClient] Fetching tools from MCP server...")
        response = await self.session.list_tools()

        gemini_tools = []
        for tool_obj in response.tools:
            try:
                input_schema_dict = tool_obj.inputSchema
                if not isinstance(input_schema_dict, dict):
                    input_schema_dict = {}

                cleaned_input_schema = input_schema_dict.copy()
                unsupported_keywords = ['additionalProperties', '$schema']
                for keyword in unsupported_keywords:
                    if keyword in cleaned_input_schema:
                        del cleaned_input_schema[keyword]

                if "type" not in cleaned_input_schema:
                    cleaned_input_schema["type"] = "object"
                if "properties" not in cleaned_input_schema:
                    cleaned_input_schema["properties"] = {}
                if "required" not in cleaned_input_schema:
                    cleaned_input_schema["required"] = []

                gemini_tools.append(
                    types.FunctionDeclaration(
                        name=tool_obj.name,
                        description=tool_obj.description,
                        parameters=cleaned_input_schema,
                    )
                )
            except AttributeError as e:
                print(f"Warning: MCP tool object missing expected attribute: {e}. Tool: {tool_obj}")
            except Exception as e:
                print(f"Error converting MCP tool: {e}")

        print(f"[MCPClient] Converted {len(gemini_tools)} tools for Gemini.")
        self.dynamic_tools_loaded = gemini_tools
        return gemini_tools

    async def process_query(self, query: str) -> str:
        if not self.session:
            raise RuntimeError("MCP session is not connected. Please connect first.")

        func_decls = await self._get_gemini_function_declarations()

        # --- IMPORTANT: Convert FunctionDeclarations to Tools for Gemini ---
        tools = [types.Tool(function_declarations=[fd]) for fd in func_decls]
        config = types.GenerateContentConfig(tools=tools)

        print(f"\n--- Processing Query: '{query}' ---")

        try:
            # --- Non-streaming (single response) ---
            response = self.genai_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[query],
                config=config,
            )
            # response.text is the final text output
            full_response_text = response.text

            # --- Tool call detection ---
            tool_calls_made = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        tool_calls_made.append(part.function_call)

            print("AI: ", end="", flush=True)
            print(full_response_text)

            if tool_calls_made:
                print(f"[MODEL] Detected tool calls: {[tc.name for tc in tool_calls_made]}")
                tool_results_to_send = []

                for tool_call in tool_calls_made:
                    tool_name = tool_call.name
                    tool_args = dict(tool_call.args)
                    try:
                        tool_result_obj = await self.session.call_tool(tool_name, tool_args)
                        if isinstance(tool_result_obj.content, mcp_common_types.TextContent):
                            result_content = tool_result_obj.content.text
                        elif isinstance(tool_result_obj.content, list):
                            # Concatenate list of TextContent
                            result_content = " ".join(getattr(tc, 'text', str(tc)) for tc in tool_result_obj.content)
                        else:
                            # Try to extract 'text' or 'value' as a fallback, or stringify
                            result_content = str(tool_result_obj.content)

                            # For debugging, also log the type and content for troubleshooting:
                            print(f"[TOOL DEBUG] Got unhandled content type: {type(tool_result_obj.content)}")
                            print(f"[TOOL DEBUG] Content: {tool_result_obj.content}")

                        tool_results_to_send.append(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"result": result_content}
                            )
                        )
                        print(f"[TOOL RESULT] {tool_name} returned: {result_content}")
                    except Exception as tool_error:
                        error_message = f"Error executing tool {tool_name}: {tool_error}"
                        print(f"[TOOL ERROR] {error_message}")
                        tool_results_to_send.append(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"error": error_message}
                            )
                        )

                print("[MODEL] Sending tool results back to model...")
                follow_up_response = self.genai_client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=tool_results_to_send,
                    config=config,
                )
                final_response = follow_up_response.text
                print("AI (after tool):", final_response)
                return final_response
            else:
                return full_response_text

        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            return f"Error: Could not process query. {e}"

    async def chat_loop(self):
        print("\n" + "=" * 50)
        print("MCP Gemini Client Started!")
        print("Type your queries or 'quit' to exit.")
        print("=" * 50 + "\n")

        while True:
            try:
                query = input("You: ").strip()
                if query.lower() == 'quit':
                    break
                result = await self.process_query(query)
                if isinstance(result, str) and result.strip():
                    print(f"AI (final): {result}")
            except Exception as e:
                print(f"\nError in chat loop: {str(e)}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_command>")
        sys.exit(1)

    client = MCPClient()
    server_command_parts = sys.argv[1].split()
    server_executable = server_command_parts[0]
    server_args = server_command_parts[1:]

    try:
        await client.connect_to_mcp_server(server_executable, *server_args)
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")
        sys.exit(1)

    try:
        await client._get_gemini_function_declarations()
    except Exception as e:
        print(f"Error initializing tools: {e}")
        print("Continuing without full AI tool capabilities.")

    try:
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
