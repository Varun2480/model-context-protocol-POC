import google.generativeai as genai
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini API with your API key
# It's recommended to set GOOGLE_API_KEY as an environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)

async def chat_loop():
    """
    Runs an asynchronous chat loop with the Gemini 1.5 Flash model.
    """
    print("Connecting to Gemini 1.5 Flash...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Start a chat session. The 'history' will be automatically managed by the chat object.
        chat = model.start_chat(history=[])
        print("Chat started! Type 'exit' to end the conversation.")
        print("--- Current Chat History (Managed by AI) ---")
        print("--------------------------------------------")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Exiting chat. Goodbye!")
                break

            try:
                print("AI: ", end="", flush=True) # Prepare for streaming output
                # Send message asynchronously with streaming enabled
                response_stream = await chat.send_message_async(user_input, stream=True)

                # Iterate over the streamed parts of the response
                full_response_text = ""
                async for chunk in response_stream:
                    if chunk.text: # Ensure the chunk has text content
                        print(chunk.text, end="", flush=True)
                        full_response_text += chunk.text
                print() # Newline after the full AI response

                # The chat object's history is automatically updated by send_message_async
                # You can inspect it if needed, but it's handled internally for the conversation.
                # For example, to see raw history:
                # print("\n--- Debugging: Raw Chat History ---")
                # for message in chat.history:
                #     print(f"{message.role}: {message.parts[0].text if message.parts else ''}")
                # print("----------------------------------\n")

            except Exception as e:
                print(f"\nError communicating with AI: {e}")
                print("Please try again or check your network connection.")

    except Exception as e:
        print(f"Failed to initialize Gemini model: {e}")
        print("Ensure your API key is valid and the model name is correct.")

if __name__ == "__main__":
    # Run the asynchronous chat loop
    asyncio.run(chat_loop())