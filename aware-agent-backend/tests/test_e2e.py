import pytest
import requests
from src.app import create_app
from src.config import TestConfig

@pytest.fixture
def app():
    app = create_app(TestConfig)
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_api_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {'status': 'healthy'}

def test_api_endpoints(client):
    """Test basic API endpoints"""
    # Add your API endpoint tests here
    # Example:
    # response = client.get('/api/endpoint')
    # assert response.status_code == 200
    pass 