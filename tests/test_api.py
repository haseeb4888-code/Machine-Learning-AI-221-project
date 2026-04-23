"""Tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check and status endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct response"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_available" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns performance data"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "regression_r2" in data
        assert "regression_rmse" in data
        assert "classification_accuracy" in data
        assert "total_countries" in data
        assert "models_trained" in data


class TestModelManagement:
    """Test model listing and management endpoints"""
    
    def test_list_models(self):
        """Test listing available models"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "regression_models" in data
        assert "classification_models" in data
        assert "total_models" in data
    
    def test_docs_schema(self):
        """Test API schema documentation"""
        response = client.get("/docs-schema")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "endpoints" in data
        assert "/health" in data["endpoints"]
