import gradio as gr
import requests
import json
import threading
import time
import os
import logging # Added for logging
from typing import Dict, List, Tuple

# --- Logging Configuration ---
# Configure logging to output to stdout, which is standard for containers
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class ChatInterface:
    def __init__(self):
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
        self.provider_status = {name: "🔴" for name in self.providers.keys()}
        self.ollama_models = []
        self.selected_model = ""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if self.ollama_base_url.endswith('/'):
            self.ollama_base_url = self.ollama_base_url[:-1]
        logger.info(f"ChatInterface initialized. Ollama base URL: {self.ollama_base_url}")

    def check_provider_status(self, provider_name: str, url: str) -> str:
        logger.info(f"Checking status for provider: {provider_name} at URL: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, timeout=3, headers=headers)
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
            return "❓" # Unknown status

    def update_all_provider_status(self):
        logger.info("Starting to update status for all providers.")
        def update_single_provider(name, url):
            self.provider_status[name] = self.check_provider_status(name, url)
        threads = []
        for name, url in self.providers.items():
            thread = threading.Thread(target=update_single_provider, args=(name, url))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        logger.info("Finished updating status for all providers.")

    def get_ollama_models(self) -> List[str]:
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
        except requests.exceptions.Timeout:
            logger.error("Connection Timeout while fetching Ollama models.")
            return ["Connection Timeout - Is Ollama running and accessible?"]
        except requests.exceptions.ConnectionError:
            logger.error("Connection Error while fetching Ollama models.")
            return ["Connection Error - Is Ollama running and accessible?"]
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error fetching Ollama models: {e.response.status_code} - {e.response.text}")
            return [f"HTTP Error: {e.response.status_code} - Check Ollama logs."]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Ollama when fetching models: {e}")
            return ["Error: Invalid JSON response from Ollama."]
        except Exception as e:
            logger.error(f"Unexpected error fetching Ollama models: {str(e)}")
            return [f"Error fetching models: {str(e)}"]

    def chat_with_ollama(self, message: str, history: List, model: str) -> List[Tuple[str, str]]:
        logger.info(f"Attempting to chat with Ollama model: {model}")
        if not model or "Error" in model or "No models found" in model or "Connection Timeout" in model:
            logger.warning(f"Cannot chat: Invalid Ollama model selected or model not accessible ('{model}').")
            return history + [(message, "Please select a valid and accessible Ollama model first.")]
        try:
            payload = {
                "model": model,
                "prompt": message,
                "stream": False
            }
            logger.debug(f"Sending payload to Ollama /api/generate: {payload}")
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60 # Increased timeout
            )
            response.raise_for_status()
            response_data = response.json()
            reply = response_data.get('response', 'No response received or unexpected format.')
            logger.info(f"Received reply from Ollama model {model}.")
            logger.debug(f"Ollama reply content: {reply[:100]}...") # Log first 100 chars of reply
            return history + [(message, reply)]
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out for model {model}.")
            return history + [(message, "Error: Ollama request timed out.")]
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama for model {model}.")
            return history + [(message, "Error: Failed to connect to Ollama. Is it running?")]
        except requests.exceptions.HTTPError as e:
            error_message_detail = ""
            try:
                error_detail_json = e.response.json()
                error_message_detail = error_detail_json.get("error", e.response.text)
            except json.JSONDecodeError:
                error_message_detail = e.response.text
            logger.error(f"Ollama API request failed for model {model} with status {e.response.status_code}. Details: {error_message_detail}")
            error_message = f"Error: Ollama API request failed with status {e.response.status_code}."
            if error_message_detail:
                 error_message += f" Details: {error_message_detail}"
            return history + [(message, error_message)]
        except json.JSONDecodeError as e:
            logger.error(f"Could not decode Ollama's response (invalid JSON) for model {model}: {e}")
            return history + [(message, "Error: Could not decode Ollama's response (invalid JSON).")]
        except Exception as e:
            logger.error(f"An unexpected error occurred while chatting with Ollama model {model}: {str(e)}")
            return history + [(message, f"An unexpected error occurred: {str(e)}")]

    def refresh_providers(self) -> gr.HTML:
        logger.info("Refreshing provider status display.")
        self.update_all_provider_status()
        html_content = "<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 10px;'><div style='color: white; font-size: 14px; line-height: 1.8;'>"
        for name, status in self.provider_status.items():
            html_content += f"{status} {name}<br/>"
        html_content += "</div></div>"
        return gr.HTML(value=html_content)

    def refresh_ollama_models(self) -> gr.Dropdown:
        logger.info("Refreshing Ollama models dropdown.")
        self.ollama_models = self.get_ollama_models()
        # Ensure the value is valid or reset if not
        current_value = self.selected_model if self.selected_model in self.ollama_models else (self.ollama_models[0] if self.ollama_models and "Error" not in self.ollama_models[0] and "No models" not in self.ollama_models[0] else "")
        logger.info(f"Ollama models dropdown refreshed. Choices: {self.ollama_models}, Selected: {current_value}")
        # The warning "The value passed into gr.Dropdown() is not in the list of choices" can occur if current_value is ""
        # and "" is not explicitly in choices. allow_custom_value=True helps mitigate this.
        return gr.Dropdown(choices=self.ollama_models, label="🤖 Ollama Model", value=current_value, allow_custom_value=True)

def create_interface():
    logger.info("Creating Gradio interface.")
    chat_instance = ChatInterface()
    css = """
    /* CSS remains the same as provided */
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        background: linear-gradient(145deg, rgba(255,255,255,0.25), rgba(255,255,255,0.1));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .chat-container {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1), 0 4px 20px rgba(0,0,0,0.15);
    }
    .model-selector {
        background: linear-gradient(145deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    .status-indicator {
        font-size: 14px;
        font-weight: 600;
        color: #2c3e50;
    }
    button {
        background: linear-gradient(145deg, #4facfe 0%, #00f2fe 100%);
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        transition: all 0.3s ease;
        font-weight: 600;
        color: white;
    }
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
    }
    .refresh-btn {
        background: linear-gradient(145deg, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
    }
    """
    with gr.Blocks(css=css, title="🚀 Ollama Chat Interface", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div class="main-header">
            <h1 style="text-align: center; color: white; font-size: 2.5em; margin: 0;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: 300;">
                🚀 Ollama Chat Interface
            </h1>
            <p style="text-align: center; color: rgba(255,255,255,0.9); font-size: 1.1em;
                        margin: 10px 0 0 0;">
                Professional AI Chat with Provider Status Monitoring
            </p>
        </div>
        """)
        with gr.Row():
            with gr.Column(scale=1, elem_classes="model-selector"):
                gr.HTML("<h3 style='color: white; text-align: center; margin-top: 0;'>🌐 Provider Status</h3>")
                provider_status_html = gr.HTML(
                    value="<!-- Initial status will be loaded by interface.load -->",
                    label="Provider Status"
                )
                refresh_providers_btn = gr.Button(
                    "🔄 Refresh Provider Status",
                    elem_classes="refresh-btn"
                )
                gr.HTML("<hr style='border-color: rgba(255,255,255,0.3); margin: 20px 0;'>")
                gr.HTML("<h3 style='color: white; text-align: center;'>🤖 Ollama Settings</h3>")
                selected_model_state = gr.State(value="")
                model_dropdown = gr.Dropdown(
                    choices=["Loading models..."], # Initial choices
                    label="Select Ollama Model",
                    value="", # Initial value
                    allow_custom_value=True # Important for handling dynamic choices and empty initial value
                )
                refresh_models_btn = gr.Button(
                    "🔄 Refresh Ollama Models",
                    elem_classes="refresh-btn"
                )
            with gr.Column(scale=3, elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    label="💬 Chat",
                    height=500,
                    show_label=True,
                    container=True,
                    # bubble_full_width=False, # Removed deprecated parameter
                    type='messages' # Added type parameter
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 🚀", scale=1)
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # Event handlers
        def handle_send_message(message_text: str, history_list: List, current_selected_model: str):
            logger.debug(f"handle_send_message called. Message: '{message_text}', Model: '{current_selected_model}'")
            if not message_text.strip():
                logger.info("Empty message received, not sending.")
                return history_list, "" # Return original history and clear input

            # Gradio's chatbot with type='messages' expects history as a list of dictionaries
            # For simplicity in chat_with_ollama, we can convert internally or adjust chat_with_ollama.
            # Here, we'll adapt the input to chat_with_ollama if needed.
            # However, the current chat_with_ollama returns a list of tuples.
            # Let's assume for now the history_list is compatible or adapt it.
            # For type='messages', history is usually:
            # e.g. [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
            # Our chat_with_ollama currently appends (user_msg, bot_msg) tuples.
            # We need to ensure consistency or convert.
            # For now, let's assume history_list is managed correctly by Gradio and what chat_with_ollama expects.

            if not current_selected_model:
                logger.warning("No Ollama model selected. Adding error to chat.")
                # Adapt to type='messages' if chatbot expects it.
                # For now, keeping tuple format for simplicity, assuming Gradio handles it.
                # If type='messages' is strictly enforced by Gradio for history, this needs adjustment.
                # The log "You have not specified a value for the `type` parameter" suggests it defaults to tuples.
                # Now that we set type='messages', Gradio will expect a list of dicts.
                # Let's adjust how we add to history for this error message.
                # The send_message function in Gradio with type='messages' for chatbot
                # should return a list of lists/tuples like [[user_msg, bot_msg], ...] or list of dicts.
                # Let's stick to list of lists/tuples for now as the original code did.
                # Gradio handles this conversion for display.
                updated_history = history_list + [[message_text, "⚠️ Please select an Ollama model first."]]
                return updated_history, message_text

            # Call chat_with_ollama with the selected model from state
            # chat_with_ollama returns history + [(message, reply)]
            new_history = chat_instance.chat_with_ollama(message_text, history_list, current_selected_model)
            return new_history, "" # Clear input box after sending

        def handle_clear_chat():
            logger.info("Clearing chat history.")
            return [], ""

        def handle_refresh_providers():
            logger.info("Handling refresh providers button click.")
            return chat_instance.refresh_providers()

        def handle_refresh_models():
            logger.info("Handling refresh Ollama models button click.")
            updated_dropdown_component = chat_instance.refresh_ollama_models()
            # The component itself is returned to update the UI, and the new selected value for the state.
            new_selected_value = chat_instance.selected_model
            logger.debug(f"Refreshed models. New selected value for state: {new_selected_value}")
            return updated_dropdown_component, new_selected_value

        def handle_model_selection(selected_model_from_dropdown: str):
            logger.info(f"Ollama model selection changed to: {selected_model_from_dropdown}")
            chat_instance.selected_model = selected_model_from_dropdown
            return selected_model_from_dropdown # Update the state

        # Connect events
        send_btn.click(
            handle_send_message,
            inputs=[msg_input, chatbot, selected_model_state],
            outputs=[chatbot, msg_input]
        )
        msg_input.submit(
            handle_send_message,
            inputs=[msg_input, chatbot, selected_model_state],
            outputs=[chatbot, msg_input]
        )
        clear_btn.click(handle_clear_chat, outputs=[chatbot, msg_input])
        refresh_providers_btn.click(
            handle_refresh_providers,
            outputs=[provider_status_html]
        )
        refresh_models_btn.click(
            handle_refresh_models,
            outputs=[model_dropdown, selected_model_state]
        )
        model_dropdown.change(
            fn=handle_model_selection,
            inputs=[model_dropdown],
            outputs=[selected_model_state]
        )

        def initial_load():
            logger.info("Performing initial load of interface data.")
            provider_html_val = chat_instance.refresh_providers()
            initial_ollama_models_list = chat_instance.get_ollama_models()
            chat_instance.ollama_models = initial_ollama_models_list
            
            initial_selected_model = ""
            if initial_ollama_models_list and isinstance(initial_ollama_models_list, list) and \
               not any(err_msg in initial_ollama_models_list[0] for err_msg in ["Error", "No models", "Connection Timeout"]):
                initial_selected_model = initial_ollama_models_list[0]
            
            chat_instance.selected_model = initial_selected_model
            logger.info(f"Initial load complete. Provider HTML ready. Ollama models: {initial_ollama_models_list}, Selected: {initial_selected_model}")
            
            return (
                provider_html_val,
                gr.Dropdown(choices=initial_ollama_models_list, value=initial_selected_model, label="🤖 Ollama Model", allow_custom_value=True),
                initial_selected_model
            )
        interface.load(
            fn=initial_load,
            outputs=[provider_status_html, model_dropdown, selected_model_state]
        )
        logger.info("Gradio interface loaded and event handlers connected.")
    return interface

if __name__ == "__main__":
    logger.info("Starting Ollama Chat Interface application.")
    app_interface = create_interface()
    app_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=os.getenv("GRADIO_DEBUG", "true").lower() == "true", # Control debug via env var
        show_error=True
    )
    logger.info("Gradio application launched.")

