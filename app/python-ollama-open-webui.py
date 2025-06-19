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
    MODIFIED: Chat functionality points directly to Ollama. UI and provider
    status checks remain the same as the original version.
    """
    def __init__(self):
        # This section remains unchanged for the provider status panel
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
            "Hugging Face": "https://huggingface.co",
            # We keep this to check the status of the Open-WebUI service
            "Open WebUI": os.getenv("OPEN_WEBUI_BASE_URL", "http://localhost:8080")
        }
        self.provider_status = {name: "🔴" for name in self.providers.keys()}
        
        # --- This section now points to Ollama for chat and models ---
        self.ollama_models = []
        self.selected_model = ""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if self.ollama_base_url.endswith('/'):
            self.ollama_base_url = self.ollama_base_url[:-1]
        
        logger.info(f"ChatInterface initialized. Ollama URL: {self.ollama_base_url}, Open-WebUI URL for status: {self.providers['Open WebUI']}")

    # MODIFIED: This function now gets models directly from Ollama
    def get_ollama_models(self) -> List[str]:
        """
        Fetches the list of available models from the Ollama /api/tags endpoint.
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
            self.ollama_models = models
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

    # MODIFIED: This function now sends chat messages directly to Ollama
    def chat_with_ollama(self, messages: List[Dict[str, str]], model: str) -> str:
        """
        Sends a conversation history to the Ollama /api/chat endpoint and returns the bot's reply.
        """
        logger.info(f"Attempting to chat with Ollama model: {model}")
        try:
            payload = { "model": model, "messages": messages, "stream": False }
            logger.debug(f"Sending payload to Ollama /api/chat: {payload}")
            response = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            reply = response_data.get('message', {}).get('content', 'Error: Unexpected response format from Ollama.')
            logger.info(f"Received reply from Ollama model {model}.")
            return reply
        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with Ollama: {type(e).__name__}"
            logger.error(f"{error_message} for model {model}.")
            return error_message
        except Exception as e:
            logger.error(f"An unexpected error occurred while chatting with Ollama model {model}: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

    # --- Provider status functions and HTML generation are UNCHANGED from original ---
    def check_provider_status(self, provider_name: str, url: str) -> str:
        logger.info(f"Checking status for provider: {provider_name} at URL: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=3, headers=headers)
            if response.status_code in [200, 401, 403]: return "🟢"
            else: return "🔴"
        except Exception: return "❓"

    def update_all_provider_status(self):
        threads = []
        for name, url in self.providers.items():
            thread = threading.Thread(target=lambda p_name=name, p_url=url: self.provider_status.update({p_name: self.check_provider_status(p_name, p_url)}))
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()

    def refresh_providers(self) -> gr.HTML:
        logger.info("Refreshing provider status display.")
        self.update_all_provider_status()
        html_content = "<div style='background: #0c322c; padding: 15px; border-radius: 10px; margin-bottom: 10px;'><div style='color: #efefef; font-size: 14px; line-height: 1.8;'>"
        for name, status in self.provider_status.items():
            html_content += f"{status} {name}<br/>"
        html_content += "</div></div>"
        return gr.HTML(value=html_content)

    def refresh_ollama_models(self) -> gr.Dropdown:
        logger.info("Refreshing Ollama models dropdown.")
        self.ollama_models = self.get_ollama_models()
        current_value = self.selected_model if self.selected_model in self.ollama_models else (self.ollama_models[0] if self.ollama_models and "Error" not in self.ollama_models[0] and "No models" not in self.ollama_models[0] else "")
        self.selected_model = current_value
        logger.info(f"Ollama models dropdown refreshed. Choices: {self.ollama_models}, Selected: {current_value}")
        return gr.Dropdown(choices=self.ollama_models, label="🤖 Ollama Model", value=current_value, allow_custom_value=True)

def create_interface():
    logger.info("Creating Gradio interface.")
    chat_instance = ChatInterface()
    
    # CORRECTED: The triple-quoted string for CSS is now properly terminated.
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

        def handle_send_message(message: str, history: List[List[str]], model: str):
            if not message.strip(): return history, ""
            history.append([message, None])
            if not model or any(err in model for err in ["Error", "No models", "Connection"]):
                history[-1][1] = "⚠️ Please select a valid Ollama model first."
                return history, ""
            messages_for_api = [{"role": "user" if i % 2 == 0 else "assistant", "content": turn[0]} for i, turn in enumerate(history)]
            messages_for_api[-1]["content"] = message
            
            # MODIFIED: Call the direct-to-ollama chat function
            bot_reply = chat_instance.chat_with_ollama(messages_for_api, model)
            history[-1][1] = bot_reply
            return history, ""

        def handle_clear_chat(): return [], ""

        # MODIFIED: Wire up components to the correct, ollama-direct functions
        send_btn.click(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        msg_input.submit(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        clear_btn.click(handle_clear_chat, outputs=[chatbot, msg_input])
        refresh_providers_btn.click(chat_instance.refresh_providers, outputs=[provider_status_html])
        refresh_models_btn.click(chat_instance.refresh_ollama_models, outputs=[model_dropdown])

        def initial_load():
            provider_html_val = chat_instance.refresh_providers()
            model_dropdown_val = chat_instance.refresh_ollama_models()
            return provider_html_val, model_dropdown_val
            
        interface.load(fn=initial_load, outputs=[provider_status_html, model_dropdown])
    return interface

if __name__ == "__main__":
    logger.info("Starting Chat Interface application.")
    app_interface = create_interface()
    app_interface.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
