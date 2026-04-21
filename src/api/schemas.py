"""Pydantic schemas for API request/response validation"""

from pydantic import BaseModel
from typing import Optional, List


class CountryFeatures(BaseModel):
    """Input features for country economic analysis"""
    population: float
    area: float
    pop_density: float
    coastline: float
    net_migration: float
    infant_mortality: float
    gdp_per_capita: float
    literacy: float
    phones_per_1000: float
    arable: float
    crops: float
    other: float
    climate: float
    birthrate: float
    deathrate: float
    agriculture: float
    industry: float
    service: float
    region: str


class PredictionResponse(BaseModel):
    """API response for predictions"""
    predicted_value: Optional[float] = None
    predicted_category: Optional[str] = None
    confidence: float
    model_used: str
    input_features: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_available: int


class MetricsResponse(BaseModel):
    """Model performance metrics response"""
    regression_r2: float
    regression_rmse: float
    classification_accuracy: float
    total_countries: int
    total_features: int
    models_trained: int
