import gradio as gr
import requests
import subprocess
import json
import threading
import time
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
        
    def check_provider_status(self, provider_name: str, url: str) -> str:
        """Check if a provider is accessible by pinging their website"""
        try:
            response = requests.get(url, timeout=3)
            return "🟢" if response.status_code in [200, 401, 403] else "🔴"
        except:
            return "🔴"
            
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
    
    def get_ollama_models(self):
        """Get available Ollama models using curl"""
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                models = [model['name'] for model in data.get('models', [])]
                return models
            else:
                return ["Connection Error - Is Ollama running?"]
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def chat_with_ollama(self, message: str, history: List, model: str):
        """Send message to Ollama using curl"""
        if not model or "Error" in model:
            return history + [(message, "Please select a valid Ollama model first.")]
            
        try:
            # Prepare the request payload
            payload = {
                "model": model,
                "prompt": message,
                "stream": False
            }
            
            # Use curl to send request to Ollama with proper encoding
            result = subprocess.run([
                "curl", "-s", "-X", "POST",
                "http://localhost:11434/api/generate",
                "-H", "Content-Type: application/json; charset=utf-8",
                "-d", json.dumps(payload, ensure_ascii=False)
            ], capture_output=True, text=True, timeout=30, encoding='utf-8')
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                reply = response.get('response', 'No response received')
                # Ensure proper Unicode handling
                if isinstance(reply, str):
                    reply = reply.encode('utf-8').decode('utf-8')
                return history + [(message, reply)]
            else:
                return history + [(message, "Error: Failed to connect to Ollama")]
                
        except Exception as e:
            return history + [(message, f"Error: {str(e)}")]
    
    def refresh_providers(self):
        """Refresh provider status and return updated HTML display"""
        self.update_all_provider_status()
        
        # Create HTML display showing all providers
        html_content = "<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 10px;'><div style='color: white; font-size: 14px; line-height: 1.8;'>"
        
        for name, status in self.provider_status.items():
            html_content += f"{status} {name}<br/>"
        
        html_content += "</div></div>"
        
        print(f"Provider status HTML: {html_content}")  # Debug
        
        return gr.HTML(value=html_content)
    
    def refresh_ollama_models(self):
        """Refresh Ollama models and return updated dropdown"""
        self.ollama_models = self.get_ollama_models()
        return gr.Dropdown(choices=self.ollama_models, label="🤖 Ollama Model", value="")

def create_interface():
    chat = ChatInterface()
    
    # Custom CSS for Frutiger Aero skeumorphic design
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
                
                # Status display for all providers
                provider_status_html = gr.HTML(
                    value="""
                    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                        <div style='color: white; font-size: 14px; line-height: 1.6;'>
                            🔄 OpenAI<br/>
                            🔄 Claude (Anthropic)<br/>
                            🔄 DeepSeek<br/>
                            🔄 Google Gemini<br/>
                            🔄 Cohere<br/>
                            🔄 Mistral AI<br/>
                            🔄 Perplexity<br/>
                            🔄 Together AI<br/>
                            🔄 Groq<br/>
                            🔄 Hugging Face
                        </div>
                    </div>
                    """,
                    label="Provider Status"
                )
                
                provider_dropdown = gr.Dropdown(
                    choices=[
                        "🔄 OpenAI",
                        "🔄 Claude (Anthropic)", 
                        "🔄 DeepSeek",
                        "🔄 Google Gemini",
                        "🔄 Cohere",
                        "🔄 Mistral AI",
                        "🔄 Perplexity",
                        "🔄 Together AI",
                        "🔄 Groq",
                        "🔄 Hugging Face"
                    ],
                    label="Select Provider (Demo)",
                    value="🔄 OpenAI",
                    interactive=True,
                    elem_classes="status-indicator",
                    multiselect=False,
                    allow_custom_value=False,
                    visible=False  # Hide this for now, we'll use the HTML display
                )
                
                refresh_providers_btn = gr.Button(
                    "🔄 Refresh Provider Status", 
                    elem_classes="refresh-btn"
                )
                
                gr.HTML("<hr style='border-color: rgba(255,255,255,0.3); margin: 20px 0;'>")
                gr.HTML("<h3 style='color: white; text-align: center;'>🤖 Ollama Settings</h3>")
                
                model_dropdown = gr.Dropdown(
                    choices=["Loading models..."],
                    label="Select Ollama Model",
                    value="",
                    allow_custom_value=True
                )
                
                refresh_models_btn = gr.Button(
                    "🔄 Refresh Ollama Models",
                    elem_classes="refresh-btn"
                )
                
                model_input = gr.Textbox(
                    label="Or Enter Model Name",
                    placeholder="e.g., llama2, mistral, codellama...",
                    value=""
                )
            
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
        def send_message(message, history, model_dropdown_val, model_input_val):
            if not message.strip():
                return history, ""
            
            # Use dropdown selection or manual input
            selected_model = model_dropdown_val if model_dropdown_val else model_input_val
            
            if not selected_model:
                return history + [(message, "⚠️ Please select or enter an Ollama model first.")], ""
            
            new_history = chat.chat_with_ollama(message, history, selected_model)
            return new_history, ""
        
        def clear_chat():
            return []
        
        def update_model_from_input(model_input_val):
            return model_input_val
        
        # Connect events
        send_btn.click(
            send_message,
            inputs=[msg_input, chatbot, model_dropdown, model_input],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            send_message,
            inputs=[msg_input, chatbot, model_dropdown, model_input],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(clear_chat, outputs=[chatbot])
        
        refresh_providers_btn.click(
            chat.refresh_providers,
            outputs=[provider_status_html]
        )
        
        refresh_models_btn.click(
            chat.refresh_ollama_models,
            outputs=[model_dropdown]
        )
        
        # Auto-refresh on startup
        def initial_load():
            print("Starting initial load...")  # Debug
            
            # Update provider status first
            chat.update_all_provider_status()
            
            # Create HTML display showing all providers
            html_content = "<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 10px;'><div style='color: white; font-size: 14px; line-height: 1.8;'>"
            
            for name, status in chat.provider_status.items():
                html_content += f"{status} {name}<br/>"
            
            html_content += "</div></div>"
            
            print(f"Initial provider status: {chat.provider_status}")  # Debug
            
            # Update Ollama models
            ollama_models = chat.get_ollama_models()
            print(f"Ollama models: {ollama_models}")  # Debug
            
            return (
                gr.HTML(value=html_content),
                gr.Dropdown(choices=ollama_models, value="")
            )
        
        interface.load(
            fn=initial_load,
            outputs=[provider_status_html, model_dropdown]
        )
    
    return interface

if __name__ == "__main__":
    # Initialize and launch the interface
    app = create_interface()
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
