import requests
import json
import threading
import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS # You'll need to install this

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class NetworkStatusChecker:
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
        # Initialize status. It will be updated by the first API call.
        self.provider_status = {name: "⚪️" for name in self.providers.keys()}
        logger.info("NetworkStatusChecker initialized.")

    def check_provider_access(self, provider_name: str, url: str) -> str:
        """Checks a single provider and returns its status icon."""
        try:
            # Use a standard user-agent to avoid being blocked
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            # A short timeout is crucial for responsiveness
            response = requests.get(url, timeout=3, headers=headers)
            
            # Any successful response or common auth error means the site is reachable
            if response.status_code in [200, 401, 403]:
                logger.info(f"Provider '{provider_name}' is ACCESSIBLE (Status: {response.status_code}).")
                return "🟢" # Green Circle: Accessible
            else:
                logger.warning(f"Provider '{provider_name}' is BLOCKED or down (Status: {response.status_code}).")
                return "🔴" # Red Circle: Blocked or Error
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            logger.error(f"Provider '{provider_name}' is BLOCKED (Connection Error/Timeout).")
            return "🔴" # Red Circle: Blocked
        except Exception as e:
            logger.error(f"Unexpected error checking '{provider_name}': {e}")
            return "❓" # Question Mark: Unknown Error

    def update_all_statuses(self):
        """Uses threading to check all providers concurrently."""
        logger.info("Starting concurrent status update for all providers.")
        
        def update_task(name, url):
            self.provider_status[name] = self.check_provider_access(name, url)

        threads = [threading.Thread(target=update_task, args=(name, url)) for name, url in self.providers.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join() # Wait for all checks to complete
        
        logger.info("Finished updating all provider statuses.")
        return self.provider_status

# --- Flask App Definition ---
app = Flask(__name__)
# CORS is required to allow the browser (on Open WebUI's domain) to call our Python backend
CORS(app) 
status_checker = NetworkStatusChecker()

@app.route('/api/status', methods=['GET'])
def get_network_status():
    """API endpoint that triggers the status check and returns the results."""
    logger.info("Received request on /api/status endpoint.")
    latest_statuses = status_checker.update_all_statuses()
    return jsonify(latest_statuses)

if __name__ == "__main__":
    logger.info("Starting SUSE Security Status Backend Server.")
    # Run on port 5000, accessible from other containers/machines on the network
    app.run(host='0.0.0.0', port=5000)