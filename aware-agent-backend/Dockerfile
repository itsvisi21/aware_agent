FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application code
COPY src/ src/
COPY deploy/ deploy/
COPY tests/ tests/
COPY pytest.ini .

# Create necessary directories
RUN mkdir -p /app/backups

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8001

# Start the application
CMD ["python", "deploy/deploy.py"] 