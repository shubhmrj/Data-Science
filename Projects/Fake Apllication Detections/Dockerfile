# Dockerfile

# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by androguard
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caching optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project files
COPY . .

# Create models folder (will be populated at runtime from HF Hub)
RUN mkdir -p models

# Expose port 7860 (REQUIRED by Hugging Face Spaces)
EXPOSE 7860

# Start Flask app
CMD ["python", "app.py"]