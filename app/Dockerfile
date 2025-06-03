# Replace with your specific SUSE Python base image if available
# FROM sles15/python:3.11
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY /app/requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Consider adding --default-timeout=100 or proxy settings if behind a corporate proxy
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY /app/app.py .

# Environment variable for Ollama API URL (can be overridden at runtime)
# The default points to Ollama running on the host machine or a locally networked service.
# For Kubernetes, this would typically be a service name like 'http://ollama-service:11434'
ENV OLLAMA_BASE_URL="http://ollama.wiredquill.com:11434"

# Expose the port Gradio will run on
EXPOSE 7860

# Command to run the application
# Using unbuffered python output for better logging in containers
CMD ["python", "-u", "app.py"]