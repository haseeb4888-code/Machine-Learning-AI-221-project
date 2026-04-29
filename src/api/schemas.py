"""Pydantic schemas for API request/response validation"""

from pydantic import BaseModel
from typing import Optional, List, Dict


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


class ClusteringResponse(BaseModel):
    """API response for clustering analysis"""
    cluster_assignment: int  # Which cluster (0, 1, 2, etc.)
    cluster_name: str  # Descriptive name
    model_used: str  # Which clustering algorithm
    silhouette_score: Optional[float] = None  # Quality metric
    cluster_size: Optional[int] = None  # Number of countries in cluster
    input_features: dict


class ClusterAnalysisResponse(BaseModel):
    """Cluster analysis and characteristics"""
    model_used: str
    n_clusters: int
    silhouette_score: float
    davies_bouldin_score: float
    cluster_distribution: Dict[str, int]  # cluster_id -> count
    cluster_descriptions: Dict[str, str]  # cluster_id -> description


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_available: int
    clustering_models: int = 0


class MetricsResponse(BaseModel):
    """Model performance metrics response"""
    regression_r2: float
    regression_rmse: float
    classification_accuracy: float
    clustering_silhouette: Optional[float] = None
    total_countries: int
    total_features: int
    models_trained: int
    clustering_models_trained: int = 0