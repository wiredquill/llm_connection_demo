import gradio as gr
import requests
import json
import threading
import time
import os
import logging
from typing import Dict, List, Any

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class ChatInterface:
    """
    Manages the application state and logic for the Gradio chat interface.
    """
    def __init__(self):
        # A dictionary of provider names and their URLs for status checking.
        self.providers = {
            "OpenAI": "https://openai.com",
            "Claude (Anthropic)": "https://anthropic.com",
            "DeepSeek": "https://deepseek.com",
            "Google Gemini": "https://aistudio.google.com",
            "Cohere": "https://cohere.ai",
            "Mistral AI": "https://mistral.ai",
            "Perplexity": "https://perplexity.ai",
            "Together AI": "https://together.xyz",
            "Groq": "https://groq.com",
            "Hugging Face": "https://huggingface.co"
        }
        # Initialize all provider statuses as unchecked.
        self.provider_status = {name: "🔴" for name in self.providers.keys()}
        self.ollama_models = []
        self.selected_model = ""
        # Get the Ollama base URL from environment variables, with a fallback for local dev.
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if self.ollama_base_url.endswith('/'):
            self.ollama_base_url = self.ollama_base_url[:-1]
        logger.info(f"ChatInterface initialized. Ollama base URL: {self.ollama_base_url}")

    def check_provider_status(self, provider_name: str, url: str) -> str:
        """
        Checks the status of a single provider by making an HTTP GET request.
        Returns a status emoji.
        """
        logger.info(f"Checking status for provider: {provider_name} at URL: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            # A short timeout is used to keep the UI responsive.
            response = requests.get(url, timeout=3, headers=headers)
            # Consider any successful or auth-related status code as "accessible".
            if response.status_code in [200, 401, 403]:
                logger.info(f"Provider {provider_name} is accessible (Status: {response.status_code}). Status: 🟢")
                return "🟢"
            else:
                logger.warning(f"Provider {provider_name} returned status {response.status_code}. Status: 🔴")
                return "🔴"
        except requests.exceptions.Timeout:
            logger.error(f"Timeout checking provider {provider_name} at {url}. Status: 🔴")
            return "🔴"
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error checking provider {provider_name} at {url}. Status: 🔴")
            return "🔴"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception checking provider {provider_name} at {url}: {e}. Status: 🔴")
            return "🔴"
        except Exception as e:
            logger.error(f"Unexpected error checking provider {provider_name} at {url}: {e}. Status: ❓")
            return "❓"

    def update_all_provider_status(self):
        """
        Uses threading to update the status of all providers concurrently for speed.
        """
        logger.info("Starting to update status for all providers.")
        def update_single_provider(name, url):
            self.provider_status[name] = self.check_provider_status(name, url)
        threads = []
        for name, url in self.providers.items():
            thread = threading.Thread(target=update_single_provider, args=(name, url))
            threads.append(thread)
            thread.start()
        # Wait for all threads to complete.
        for thread in threads:
            thread.join()
        logger.info("Finished updating status for all providers.")

    def get_ollama_models(self) -> List[str]:
        """
        Fetches the list of available models from the Ollama /api/tags endpoint.
        Includes robust error handling for UI feedback.
        """
        logger.info(f"Attempting to fetch Ollama models from {self.ollama_base_url}/api/tags")
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            if not models:
                logger.warning("No Ollama models found at the endpoint.")
                return ["No models found at Ollama endpoint."]
            logger.info(f"Successfully fetched Ollama models: {models}")
            return models
        except requests.exceptions.RequestException:
            logger.error("Connection Error or Timeout while fetching Ollama models.")
            return ["Connection Error - Is Ollama running?"]
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from Ollama when fetching models.")
            return ["Error: Invalid JSON response from Ollama."]
        except Exception as e:
            logger.error(f"Unexpected error fetching Ollama models: {str(e)}")
            return [f"Error fetching models: {str(e)}"]

    def chat_with_ollama(self, messages: List[Dict[str, str]], model: str) -> str:
        """
        Sends a conversation history to the Ollama /api/chat endpoint and returns the bot's reply.
        Now takes the structured `messages` list directly.
        """
        logger.info(f"Attempting to chat with Ollama model: {model}")
        try:
            payload = {
                "model": model,
                "messages": messages, # Use the structured messages list
                "stream": False
            }
            logger.debug(f"Sending payload to Ollama /api/chat: {payload}")
            response = requests.post(
                f"{self.ollama_base_url}/api/chat", # Using the more robust /api/chat endpoint
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()
            # The response structure for /api/chat is different from /api/generate
            reply = response_data.get('message', {}).get('content', 'Error: Unexpected response format from Ollama.')
            logger.info(f"Received reply from Ollama model {model}.")
            return reply
        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with Ollama: {type(e).__name__}"
            logger.error(f"{error_message} for model {model}.")
            return error_message
        except json.JSONDecodeError:
            logger.error(f"Could not decode Ollama's response (invalid JSON) for model {model}.")
            return "Error: Could not decode Ollama's response."
        except Exception as e:
            logger.error(f"An unexpected error occurred while chatting with Ollama model {model}: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

    def refresh_providers(self) -> gr.HTML:
        """
        Updates provider statuses and returns an HTML component for display.
        """
        logger.info("Refreshing provider status display.")
        self.update_all_provider_status()
        # Using SUSE 'Pine' for the background of the status box
        html_content = "<div style='background: #0c322c; padding: 15px; border-radius: 10px; margin-bottom: 10px;'><div style='color: #efefef; font-size: 14px; line-height: 1.8;'>"
        for name, status in self.provider_status.items():
            html_content += f"{status} {name}<br/>"
        html_content += "</div></div>"
        return gr.HTML(value=html_content)

    def refresh_ollama_models(self) -> gr.Dropdown:
        """
        Refreshes the list of Ollama models and updates the dropdown component.
        """
        logger.info("Refreshing Ollama models dropdown.")
        self.ollama_models = self.get_ollama_models()
        # Set the current value to the selected model if it's still valid, otherwise the first in the list.
        current_value = self.selected_model if self.selected_model in self.ollama_models else (self.ollama_models[0] if self.ollama_models and "Error" not in self.ollama_models[0] and "No models" not in self.ollama_models[0] else "")
        self.selected_model = current_value # Update state
        logger.info(f"Ollama models dropdown refreshed. Choices: {self.ollama_models}, Selected: {current_value}")
        return gr.Dropdown(choices=self.ollama_models, label="🤖 Ollama Model", value=current_value, allow_custom_value=True)

def create_interface():
    """
    Creates and configures the Gradio interface, its components, and event handlers.
    """
    logger.info("Creating Gradio interface.")
    chat_instance = ChatInterface()
    
    css = """
    .gradio-container { background-color: #0c322c; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #efefef; }
    .main-header { background-color: #30ba78; border: 1px solid #30ba78; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); padding: 20px; margin-bottom: 20px; }
    .control-panel, .chat-container { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(48, 186, 120, 0.2); border-radius: 12px; padding: 15px; }
    h1, h2, h3 { color: #ffffff; font-weight: 600; }
    .gr-button { background: #30ba78; color: #ffffff; border: none; border-radius: 8px; box-shadow: 0 4px 10px rgba(48, 186, 120, 0.3); transition: all 0.2s ease; font-weight: 600; }
    .gr-button:hover { background: #35d489; transform: translateY(-1px); box-shadow: 0 6px 15px rgba(48, 186, 120, 0.4); }
    .refresh-btn { background: #2453ff; }
    .refresh-btn:hover { background: #4f75ff; }
    .gr-chatbot .message.user { background-color: #efefef; color: #0c322c; }
    .gr-chatbot .message.bot { background-color: #e0f8ee; color: #0c322c; }
    """
    
    with gr.Blocks(css=css, title="SUSE AI Chat", theme=gr.themes.Base()) as interface:
        gr.HTML("""
        <div class="main-header">
            <h1 style="text-align: center; color: white; font-size: 2.2em; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">SUSE AI Chat</h1>
            <p style="text-align: center; color: rgba(255,255,255,0.95); font-size: 1.1em;">Powered by Ollama | Provider Status Monitor</p>
        </div>
        """)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="control-panel"):
                gr.HTML("<h3 style='text-align: center; margin-top: 0;'>🌐 Provider Status</h3>")
                provider_status_html = gr.HTML()
                refresh_providers_btn = gr.Button("🔄 Refresh Status", elem_classes="refresh-btn")
                
                gr.HTML("<hr style='border-color: rgba(255,255,255,0.2); margin: 20px 0;'>")
                gr.HTML("<h3 style='text-align: center;'>🤖 Ollama Settings</h3>")
                model_dropdown = gr.Dropdown(choices=["Loading..."], label="Select Ollama Model", value="", allow_custom_value=True)
                refresh_models_btn = gr.Button("🔄 Refresh Models", elem_classes="refresh-btn")

            with gr.Column(scale=3):
                 with gr.Column(elem_classes="chat-container"):
                    chatbot = gr.Chatbot(label="💬 Chat", height=550, show_label=False, bubble_full_width=False, elem_id="chatbot")
                    with gr.Row():
                        msg_input = gr.Textbox(label="Message", placeholder="Type your message...", lines=2, scale=4)
                        send_btn = gr.Button("Send ❯", scale=1, variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # --- Event Handlers ---

        def handle_send_message(message: str, history: List[List[str]], model: str):
            """
            Handles the chat submission process.
            The `history` parameter is now correctly typed as `List[List[str]]` which is what Gradio provides.
            """
            logger.debug(f"Handling message: '{message}' for model: '{model}'")
            if not message.strip():
                return history, ""

            # Append the user's message to the chat history for immediate display.
            # The bot's response will be added later.
            history.append([message, None])
            
            # Check if a valid model is selected before proceeding.
            if not model or any(err in model for err in ["Error", "No models", "Connection"]):
                history[-1][1] = "⚠️ Please select a valid Ollama model first."
                return history, "" # Return updated history and clear input box.

            # Convert the Gradio chat history format `[[user, bot], ...]` to the
            # Ollama API format `[{"role": "user", "content": "..."}, ...]`
            messages_for_api = []
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages_for_api.append({"role": "user", "content": user_msg})
                if assistant_msg: # Don't add the `None` placeholder
                    messages_for_api.append({"role": "assistant", "content": assistant_msg})
            
            # Get the bot's reply from the Ollama backend.
            bot_reply = chat_instance.chat_with_ollama(messages_for_api, model)
            
            # Update the last item in the history list with the actual reply.
            history[-1][1] = bot_reply
            
            return history, ""

        def handle_clear_chat():
            logger.info("Clearing chat history.")
            return [], ""

        # Wire up components to event handlers
        send_btn.click(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        msg_input.submit(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        clear_btn.click(handle_clear_chat, outputs=[chatbot, msg_input])
        refresh_providers_btn.click(chat_instance.refresh_providers, outputs=[provider_status_html])
        refresh_models_btn.click(chat_instance.refresh_ollama_models, outputs=[model_dropdown])

        def initial_load():
            """
            Loads initial data when the interface starts up.
            """
            logger.info("Performing initial load of interface data.")
            provider_html_val = chat_instance.refresh_providers()
            model_dropdown_val = chat_instance.refresh_ollama_models()
            logger.info("Initial load complete.")
            return provider_html_val, model_dropdown_val
            
        interface.load(fn=initial_load, outputs=[provider_status_html, model_dropdown])
        logger.info("Gradio interface loaded and event handlers connected.")
    
    return interface

if __name__ == "__main__":
    logger.info("Starting Ollama Chat Interface application.")
    app_interface = create_interface()
    app_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=os.getenv("GRADIO_DEBUG", "true").lower() == "true",
        show_error=True
    )
    logger.info("Gradio application launched.")
