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
    MODIFIED: Chat now talks directly to Ollama. Open-WebUI is for status checks.
    """
    def __init__(self):
        # --- Provider Status Section (includes Open-WebUI) ---
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
            "Open WebUI": os.getenv("OPEN_WEBUI_BASE_URL", "http://localhost:8080")
        }
        self.provider_status = {name: "🔴" for name in self.providers.keys()}

        # --- Direct Ollama Connection Section ---
        self.ollama_models = []
        self.selected_model = ""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        logger.info(f"ChatInterface initialized. Ollama URL: {self.ollama_base_url}, Open-WebUI URL: {self.providers['Open WebUI']}")

    def get_ollama_models(self) -> List[str]:
        """
        Fetches the list of available models directly from the Ollama /api/tags endpoint.
        """
        logger.info(f"Attempting to fetch Ollama models from {self.ollama_base_url}/api/tags")
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            if not models:
                logger.warning("No Ollama models found.")
                return ["No models found at Ollama endpoint."]
            logger.info(f"Successfully fetched Ollama models: {models}")
            self.ollama_models = models
            return models
        except requests.exceptions.RequestException:
            logger.error("Connection Error fetching Ollama models.")
            return ["Connection Error - Is Ollama running?"]
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return [f"Error: {str(e)}"]

    def chat_with_ollama(self, messages: List[Dict[str, str]], model: str) -> str:
        """
        Sends a conversation history directly to the Ollama /api/chat endpoint.
        """
        logger.info(f"Attempting to chat with Ollama model: {model}")
        try:
            payload = {"model": model, "messages": messages, "stream": False}
            response = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            reply = response_data.get('message', {}).get('content', 'Error: Unexpected response format.')
            logger.info(f"Received reply from Ollama model {model}.")
            return reply
        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with Ollama: {type(e).__name__}"
            logger.error(f"{error_message} for model {model}.")
            return error_message
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

    # --- Provider status functions remain unchanged ---
    def check_provider_status(self, provider_name: str, url: str) -> str:
        logger.info(f"Checking status for provider: {provider_name} at URL: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=3, headers=headers)
            if response.status_code in [200, 401, 403]: return "🟢"
            else: return "🔴"
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError): return "🔴"
        except Exception: return "❓"

    def update_all_provider_status(self):
        threads = []
        for name, url in self.providers.items():
            thread = threading.Thread(target=lambda p_name, p_url: setattr(self, 'provider_status', {**self.provider_status, p_name: self.check_provider_status(p_name, p_url)}), args=(name, url))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def refresh_providers_html(self) -> gr.HTML:
        logger.info("Refreshing provider status display.")
        self.update_all_provider_status()
        html_content = "<div style='padding: 15px; border-radius: 10px;'>"
        for name, status in self.provider_status.items():
            html_content += f"{status} {name}<br/>"
        html_content += "</div>"
        return gr.HTML(value=html_content)
    
    def refresh_ollama_models_dropdown(self) -> gr.Dropdown:
        logger.info("Refreshing Ollama models dropdown.")
        models = self.get_ollama_models()
        current_value = self.selected_model if self.selected_model in models else (models[0] if models and "Error" not in models[0] else "")
        self.selected_model = current_value
        return gr.Dropdown(choices=models, label="🤖 Ollama Model", value=current_value, allow_custom_value=True)

def create_interface():
    logger.info("Creating Gradio interface.")
    chat_instance = ChatInterface()
    
    with gr.Blocks(title="Ollama Chat", theme=gr.themes.Base()) as interface:
        gr.HTML("<h1>Direct to Ollama Chat</h1><p>Chat with models served by your Ollama instance.</p>")
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🌐 Provider Status</h3>")
                provider_status_html = gr.HTML()
                refresh_providers_btn = gr.Button("🔄 Refresh Status")
                
                gr.HTML("<hr><h3>🤖 Ollama Settings</h3>")
                model_dropdown = gr.Dropdown(choices=["Loading..."], label="Select Ollama Model")
                refresh_models_btn = gr.Button("🔄 Refresh Models")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=550)
                with gr.Row():
                    msg_input = gr.Textbox(label="Message", placeholder="Type your message...", lines=2, scale=4)
                    send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat")

        def handle_send_message(message: str, history: List[List[str]], model: str):
            if not message.strip(): return history, ""
            history.append([message, None])
            if not model or "Error" in model or "No models" in model:
                history[-1][1] = "⚠️ Please select a valid Ollama model first."
                return history, ""
            messages_for_api = [{"role": "user" if i % 2 == 0 else "assistant", "content": turn[1]} for i, turn in enumerate(history) if turn[1]]
            messages_for_api.insert(-1, {"role": "user", "content": message})
            
            bot_reply = chat_instance.chat_with_ollama(messages_for_api, model)
            history[-1][1] = bot_reply
            return history, ""
        
        send_btn.click(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        msg_input.submit(handle_send_message, inputs=[msg_input, chatbot, model_dropdown], outputs=[chatbot, msg_input])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])
        refresh_providers_btn.click(chat_instance.refresh_providers_html, outputs=[provider_status_html])
        refresh_models_btn.click(chat_instance.refresh_ollama_models_dropdown, outputs=[model_dropdown])
        
        def initial_load():
            provider_html_val = chat_instance.refresh_providers_html()
            model_dropdown_val = chat_instance.refresh_ollama_models_dropdown()
            return provider_html_val, model_dropdown_val
            
        interface.load(fn=initial_load, outputs=[provider_status_html, model_dropdown])
    return interface

if __name__ == "__main__":
    app_interface = create_interface()
    app_interface.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

