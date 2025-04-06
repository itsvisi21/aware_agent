#!/bin/bash

# Deployment script for Aware Agent

# Load environment variables
source .env

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."
if ! command_exists python3; then
    echo "Python 3 is required but not installed"
    exit 1
fi

if ! command_exists pip; then
    echo "pip is required but not installed"
    exit 1
fi

# Set environment
ENVIRONMENT=${1:-development}
export ENVIRONMENT=$ENVIRONMENT

echo "Deploying to $ENVIRONMENT environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run database migrations
echo "Running database migrations..."
python manage.py migrate

# Configure logging
echo "Configuring logging..."
mkdir -p logs
touch logs/app.log

# Start the application
echo "Starting the application..."
if [ "$ENVIRONMENT" = "production" ]; then
    # Production deployment with gunicorn
    if ! command_exists gunicorn; then
        pip install gunicorn
    fi
    gunicorn -w 4 -b 0.0.0.0:${WEBSOCKET_PORT:-8000} app:app --access-logfile logs/access.log --error-logfile logs/error.log
else
    # Development deployment
    python app.py
fi 