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
    MODIFIED: Now loads configuration from config.json and includes auto-refresh for providers.
    """
    def __init__(self):
        self.config_path = 'config.json'
        self.config = self.load_or_create_config()

        # Provider status is now loaded from config
        self.provider_status = {name: "🔴" for name in self.config.get('providers', {}).keys()}
        
        # Add Open WebUI to the provider list for status checking, if it's defined
        webui_url = os.getenv("OPEN_WEBUI_BASE_URL")
        if webui_url:
            self.config.setdefault('providers', {})['Open WebUI'] = webui_url
        
        # --- Ollama connection section remains unchanged ---
        self.ollama_models = []
        self.selected_model = ""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if self.ollama_base_url.endswith('/'):
            self.ollama_base_url = self.ollama_base_url[:-1]
        
        logger.info(f"ChatInterface initialized. Ollama URL: {self.ollama_base_url}")
        logger.info(f"Provider auto-refresh enabled: {self.config.get('auto_refresh_providers', True)}, Interval: {self.config.get('auto_refresh_interval_seconds', 60)}s")

    def load_or_create_config(self) -> Dict:
        """Loads configuration from config.json, or creates it with defaults if it doesn't exist."""
        default_config = {
            "providers": {
                "OpenAI": "https://openai.com", "Claude (Anthropic)": "https://anthropic.com",
                "DeepSeek": "https://deepseek.com", "Google Gemini": "https://aistudio.google.com",
                "Cohere": "https://cohere.ai", "Mistral AI": "https://mistral.ai",
                "Perplexity": "https://perplexity.ai", "Together AI": "https://together.xyz",
                "Groq": "https://groq.com", "Hugging Face": "https://huggingface.co"
            },
            "auto_refresh_providers": True,
            "auto_refresh_interval_seconds": 60
        }
        
        if os.path.exists(self.config_path):
            logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path, 'r') as f:
                try:
                    config_data = json.load(f)
                    # Ensure all default keys are present
                    for key, value in default_config.items():
                        config_data.setdefault(key, value)
                    return config_data
                except json.JSONDecodeError:
                    logger.error("Error decoding config.json, using default config.")
                    return default_config
        else:
            logger.info(f"Creating default configuration file at {self.config_path}")
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config

    def get_ollama_models(self) -> List[str]:
        """Fetches the list of available models from the Ollama /api/tags endpoint."""
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
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return ["Connection Error - Is Ollama running?"]

    def chat_with_ollama(self, messages: List[Dict[str, str]], model: str) -> str:
        """Sends a conversation history to the Ollama /api/chat endpoint."""
        logger.info(f"Attempting to chat with Ollama model: {model}")
        try:
            payload = { "model": model, "messages": messages, "stream": False }
            response = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            reply = response_data.get('message', {}).get('content', 'Error: Unexpected response format from Ollama.')
            return reply
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"

    def check_provider_status(self, provider_name: str, url: str) -> str:
        """Checks the status of a single provider."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=3, headers=headers)
            return "🟢" if response.status_code in [200, 401, 403] else "🔴"
        except Exception:
            return "❓"

    def update_all_provider_status(self):
        """Updates all provider statuses in the background."""
        logger.info("Updating all provider statuses.")
        threads = []
        # Use .get() to safely access 'providers'
        for name, url in self.config.get('providers', {}).items():
            thread = threading.Thread(target=lambda p_name=name, p_url=url: self.provider_status.update({p_name: self.check_provider_status(p_name, p_url)}))
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()

    def refresh_providers(self) -> gr.HTML:
        """Manually refreshes provider statuses and returns the HTML."""
        logger.info("Manual provider refresh triggered.")
        self.update_all_provider_status()
        html_content = "<div style='background: #0c322c; padding: 15px; border-radius: 10px; margin-bottom: 10px;'><div style='color: #efefef; font-size: 14px; line-height: 1.8;'>"
        # Use a temporary copy of provider status to avoid issues with dict size changes during iteration
        for name, status in list(self.provider_status.items()):
            html_content += f"{status} {name}<br/>"
        html_content += "</div></div>"
        return gr.HTML(value=html_content)

    def refresh_ollama_models(self) -> gr.Dropdown:
        """Refreshes the dropdown with models from Ollama."""
        logger.info("Refreshing Ollama models dropdown.")
        self.ollama_models = self.get_ollama_models()
        current_value = self.selected_model if self.selected_model in self.ollama_models else (self.ollama_models[0] if self.ollama_models and "Error" not in self.ollama_models[0] else "")
        self.selected_model = current_value
        return gr.Dropdown(choices=self.ollama_models, label="🤖 Ollama Model", value=current_value, allow_custom_value=True)

def create_interface():
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
                
                with gr.Row():
                    refresh_providers_btn = gr.Button("🔄 Refresh Status", elem_classes="refresh-btn", scale=2)
                    auto_refresh_toggle = gr.Checkbox(label="Auto", value=chat_instance.config.get('auto_refresh_providers', True), scale=1)
                
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
            bot_reply = chat_instance.chat_with_ollama(messages_for_api, model)
            history[-1][1] = bot_reply
            return history, ""

        send_btn.click(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        msg_input.submit(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])
        
        refresh_providers_btn.click(chat_instance.refresh_providers, outputs=[provider_status_html])
        refresh_models_btn.click(chat_instance.refresh_ollama_models, outputs=[model_dropdown])

        def auto_refresh_function(is_enabled):
            if is_enabled:
                # We need to return the updated HTML component
                return chat_instance.refresh_providers()
            # If auto-refresh is disabled, we tell Gradio to skip the update
            return gr.skip()
        
        def initial_load():
            return chat_instance.refresh_providers(), chat_instance.refresh_ollama_models()
        
        # This will run the initial_load function once when the app starts
        interface.load(initial_load, outputs=[provider_status_html, model_dropdown])
        
        # CORRECTED: This sets up the recurring event correctly. It runs on its own `load` event.
        # The `auto_refresh_function` will be called every X seconds, and its logic
        # depends on the state of the `auto_refresh_toggle` checkbox.
        interface.load(
            fn=auto_refresh_function,
            inputs=[auto_refresh_toggle],
            outputs=[provider_status_html],
            every=chat_instance.config.get('auto_refresh_interval_seconds', 60)
        )

    return interface

if __name__ == "__main__":
    logger.info("Starting Chat Interface application.")
    app_interface = create_interface()
    app_interface.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
