import gradio as gr
import requests
import json
import threading
import time
import os # Added for environment variables
from typing import Dict, List, Tuple

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
        # Get Ollama base URL from environment variable, default to localhost
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # It's good practice to ensure no trailing slash for base URL
        if self.ollama_base_url.endswith('/'):
            self.ollama_base_url = self.ollama_base_url[:-1]

    def check_provider_status(self, provider_name: str, url: str) -> str:
        """Check if a provider is accessible by pinging their website"""
        try:
            # Added a user-agent as some sites might block default requests
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, timeout=3, headers=headers)
            # Consider 200 as success. 401/403 mean accessible but auth required.
            return "🟢" if response.status_code in [200, 401, 403] else "🔴"
        except requests.exceptions.RequestException:
            return "🔴"
        except Exception: # Catch any other unexpected errors
            return "❓" # Unknown status

    def update_all_provider_status(self):
        """Update status for all providers concurrently"""
        def update_single_provider(name, url):
            self.provider_status[name] = self.check_provider_status(name, url)

        threads = []
        for name, url in self.providers.items():
            thread = threading.Thread(target=update_single_provider, args=(name, url))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def get_ollama_models(self) -> List[str]:
        """Get available Ollama models using the requests library"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            if not models:
                return ["No models found at Ollama endpoint."]
            return models
        except requests.exceptions.Timeout:
            return ["Connection Timeout - Is Ollama running and accessible?"]
        except requests.exceptions.ConnectionError:
            return ["Connection Error - Is Ollama running and accessible?"]
        except requests.exceptions.HTTPError as e:
            return [f"HTTP Error: {e.response.status_code} - Check Ollama logs."]
        except json.JSONDecodeError:
            return ["Error: Invalid JSON response from Ollama."]
        except Exception as e:
            return [f"Error fetching models: {str(e)}"]

    def chat_with_ollama(self, message: str, history: List, model: str) -> List[Tuple[str, str]]:
        """Send message to Ollama using the requests library"""
        if not model or "Error" in model or "No models found" in model or "Connection Timeout" in model:
            return history + [(message, "Please select a valid and accessible Ollama model first.")]

        try:
            payload = {
                "model": model,
                "prompt": message,
                "stream": False # Keeping stream False as in original
            }

            # Send request to Ollama
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60 # Increased timeout for potentially long generations
            )
            response.raise_for_status() # Raise an exception for HTTP errors

            response_data = response.json()
            reply = response_data.get('response', 'No response received or unexpected format.')

            return history + [(message, reply)]

        except requests.exceptions.Timeout:
            return history + [(message, "Error: Ollama request timed out.")]
        except requests.exceptions.ConnectionError:
            return history + [(message, "Error: Failed to connect to Ollama. Is it running?")]
        except requests.exceptions.HTTPError as e:
            error_message = f"Error: Ollama API request failed with status {e.response.status_code}."
            try:
                # Try to get more specific error from Ollama response
                error_detail = e.response.json().get("error", "")
                if error_detail:
                    error_message += f" Details: {error_detail}"
            except json.JSONDecodeError:
                pass # No JSON in error response
            return history + [(message, error_message)]
        except json.JSONDecodeError:
            return history + [(message, "Error: Could not decode Ollama's response (invalid JSON).")]
        except Exception as e:
            return history + [(message, f"An unexpected error occurred: {str(e)}")]

    def refresh_providers(self) -> gr.HTML:
        """Refresh provider status and return updated HTML display"""
        self.update_all_provider_status()

        html_content = "<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 10px;'><div style='color: white; font-size: 14px; line-height: 1.8;'>"

        for name, status in self.provider_status.items():
            html_content += f"{status} {name}<br/>"

        html_content += "</div></div>"

        # print(f"Provider status HTML: {html_content}") # Debug

        return gr.HTML(value=html_content)

    def refresh_ollama_models(self) -> gr.Dropdown:
        """Refresh Ollama models and return updated dropdown"""
        self.ollama_models = self.get_ollama_models()
        # Ensure the value is valid or reset if not
        current_value = self.selected_model if self.selected_model in self.ollama_models else (self.ollama_models[0] if self.ollama_models else "")
        return gr.Dropdown(choices=self.ollama_models, label="🤖 Ollama Model", value=current_value)

def create_interface():
    chat_instance = ChatInterface() # Renamed to avoid conflict with module name

    # Custom CSS for Frutiger Aero skeumorphic design (same as original)
    css = """
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
                    value="",
                    label="Provider Status"
                )

                refresh_providers_btn = gr.Button(
                    "🔄 Refresh Provider Status",
                    elem_classes="refresh-btn"
                )

                gr.HTML("<hr style='border-color: rgba(255,255,255,0.3); margin: 20px 0;'>")
                gr.HTML("<h3 style='color: white; text-align: center;'>🤖 Ollama Settings</h3>")

                # Store selected model in a Gradio State to persist selection
                selected_model_state = gr.State(value="")

                model_dropdown = gr.Dropdown(
                    choices=["Loading models..."],
                    label="Select Ollama Model",
                    value="",
                    # allow_custom_value=True # Kept from original, though direct selection is safer
                )

                refresh_models_btn = gr.Button(
                    "🔄 Refresh Ollama Models",
                    elem_classes="refresh-btn"
                )

                # Removed model_input Textbox as dropdown is primary and can be typed into if allow_custom_value=True
                # If manual input is strongly desired alongside dropdown, it can be re-added.
                # For now, simplifying to just dropdown.

            with gr.Column(scale=3, elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    label="💬 Chat",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
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
        def handle_send_message(message: str, history: List, current_selected_model: str):
            if not message.strip():
                return history, "" # Return original history and clear input

            if not current_selected_model: # Check if a model is selected from the state
                 # Add error to history, keep user message in input
                updated_history = history + [(message, "⚠️ Please select an Ollama model first.")]
                return updated_history, message # Keep message in input box

            # Call chat_with_ollama with the selected model from state
            new_history = chat_instance.chat_with_ollama(message, history, current_selected_model)
            return new_history, "" # Clear input box after sending

        def handle_clear_chat():
            return [], "" # Clear chatbot and message input

        def handle_refresh_providers():
            return chat_instance.refresh_providers()

        def handle_refresh_models():
            # This will update chat_instance.ollama_models and chat_instance.selected_model
            updated_dropdown = chat_instance.refresh_ollama_models()
            # Return the updated dropdown component itself and the new selected value for the state
            return updated_dropdown, chat_instance.selected_model

        def handle_model_selection(selected_model_from_dropdown: str):
            chat_instance.selected_model = selected_model_from_dropdown # Update instance variable
            return selected_model_from_dropdown # Update the state

        # Connect events
        send_btn.click(
            handle_send_message,
            inputs=[msg_input, chatbot, selected_model_state], # Use selected_model_state
            outputs=[chatbot, msg_input]
        )

        msg_input.submit(
            handle_send_message,
            inputs=[msg_input, chatbot, selected_model_state], # Use selected_model_state
            outputs=[chatbot, msg_input]
        )

        clear_btn.click(handle_clear_chat, outputs=[chatbot, msg_input])

        refresh_providers_btn.click(
            handle_refresh_providers, # Use the new handler
            outputs=[provider_status_html]
        )

        refresh_models_btn.click(
            handle_refresh_models, # Use the new handler
            outputs=[model_dropdown, selected_model_state] # Update dropdown and state
        )

        # When dropdown value changes, update the selected_model_state
        model_dropdown.change(
            fn=handle_model_selection,
            inputs=[model_dropdown],
            outputs=[selected_model_state]
        )

        # Auto-refresh on startup
        def initial_load():
            # print("Starting initial load...") # Debug

            # Update provider status
            provider_html_val = chat_instance.refresh_providers() # This now returns the HTML component directly

            # Update Ollama models and get the initial dropdown and selected model
            initial_ollama_models = chat_instance.get_ollama_models()
            chat_instance.ollama_models = initial_ollama_models # Store fetched models

            # Select the first model if available, otherwise empty
            initial_selected_model = initial_ollama_models[0] if initial_ollama_models and "Error" not in initial_ollama_models[0] and "No models" not in initial_ollama_models[0] else ""
            chat_instance.selected_model = initial_selected_model

            # print(f"Initial provider status HTML: {provider_html_val.value}") # Debug
            # print(f"Initial Ollama models: {initial_ollama_models}") # Debug
            # print(f"Initial selected model: {initial_selected_model}") # Debug

            return (
                provider_html_val, # Directly return the component
                gr.Dropdown(choices=initial_ollama_models, value=initial_selected_model, label="🤖 Ollama Model"),
                initial_selected_model # For the state
            )

        interface.load(
            fn=initial_load,
            outputs=[provider_status_html, model_dropdown, selected_model_state]
        )

    return interface

if __name__ == "__main__":
    # Initialize and launch the interface
    app_interface = create_interface()
    app_interface.launch(
        server_name="0.0.0.0", # Listen on all network interfaces
        server_port=7860,
        share=False, # Set to True if you want to share via Gradio link (requires internet)
        debug=True, # Enable debug mode for more detailed errors
        show_error=True
    )