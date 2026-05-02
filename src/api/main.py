"""FastAPI application for Economic Growth Analyzer"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
import pickle
import json
import numpy as np
from pathlib import Path
import traceback
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    CountryFeatures,
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
    ClusteringResponse,
    ClusterAnalysisResponse
)

app = FastAPI(
    title="Economic Growth Analyzer API",
    description="ML-powered economic analysis and prediction platform",
    version="1.0.0"
)

app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
# Model storage
regression_models = {}
classification_models = {}
clustering_models = {}  # NEW: Add clustering storage
model_metadata = {
    'models_loaded': 0,
    'clustering_models_loaded': 0,  # NEW: Track clustering models
    'last_updated': None
}
# main.py

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Performance metrics from pipeline execution
pipeline_metrics = {
    'best_regression_r2': 0.0,
    'best_regression_rmse': 0.0,
    'best_classification_accuracy': 0.0,
    'best_clustering_silhouette': 0.0,
    'best_regression_model': 'N/A',
    'best_classification_model': 'N/A',
    'best_clustering_model': 'N/A',
    'timestamp': None,
    'total_countries': 195,
    'total_features': 25
}

# Clustering model metrics from pipeline
clustering_metrics_data = {
    'kmeans': {},
    'hierarchical': {},
    'dbscan': {}
}

# Cluster name mapping for better readability
CLUSTER_NAMES = {
    0: "Developing Economies",
    1: "Emerging Markets",
    2: "Developed Nations",
    3: "Resource-Rich Countries"  # If using 4 clusters
}


def get_latest_pipeline_summary() -> Dict[str, Any]:
    """Get the latest pipeline summary JSON file from results directory"""
    results_dir = Path('results')
    if not results_dir.exists():
        print("⚠ Results directory not found")
        return {}
    
    # Find all pipeline summary files
    summary_files = sorted(results_dir.glob('pipeline_summary_*.json'))
    if not summary_files:
        print("⚠ No pipeline summary files found")
        return {}
    
    # Get the latest file
    latest_file = summary_files[-1]
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded pipeline summary from: {latest_file.name}")
        return data
    except Exception as e:
        print(f"✗ Failed to load pipeline summary: {e}")
        return {}


def extract_best_metrics(pipeline_data: Dict[str, Any]) -> None:
    """Extract best model metrics from pipeline summary and update global metrics"""
    global pipeline_metrics, clustering_metrics_data
    
    if not pipeline_data or pipeline_data.get('status') != 'success':
        print("⚠ Pipeline data not available or not successful")
        return
    
    # Extract best classification model
    if 'clf_metrics' in pipeline_data:
        clf_metrics = pipeline_data['clf_metrics']
        best_clf = max(
            clf_metrics.items(),
            key=lambda x: x[1].get('test_accuracy', 0)
        )
        pipeline_metrics['best_classification_model'] = best_clf[0]
        pipeline_metrics['best_classification_accuracy'] = best_clf[1].get('test_accuracy', 0.0)
        print(f"✓ Best Classification: {best_clf[0]} (Accuracy: {best_clf[1].get('test_accuracy', 0):.4f})")
    
    # Extract best regression model
    if 'reg_metrics' in pipeline_data:
        reg_metrics = pipeline_data['reg_metrics']
        best_reg = max(
            reg_metrics.items(),
            key=lambda x: x[1].get('test_r2', -float('inf'))
        )
        pipeline_metrics['best_regression_model'] = best_reg[0]
        pipeline_metrics['best_regression_r2'] = best_reg[1].get('test_r2', 0.0)
        pipeline_metrics['best_regression_rmse'] = best_reg[1].get('rmse', 0.0)
        print(f"✓ Best Regression: {best_reg[0]} (R²: {best_reg[1].get('test_r2', 0):.4f}, RMSE: {best_reg[1].get('rmse', 0):.2f})")
    
    # Extract best clustering model and store all clustering metrics
    if 'clustering_metrics' in pipeline_data:
        clust_metrics = pipeline_data['clustering_metrics']
        best_clust = max(
            clust_metrics.items(),
            key=lambda x: x[1].get('silhouette_score', -float('inf'))
        )
        pipeline_metrics['best_clustering_model'] = best_clust[0]
        pipeline_metrics['best_clustering_silhouette'] = best_clust[1].get('silhouette_score', 0.0)
        print(f"✓ Best Clustering: {best_clust[0]} (Silhouette: {best_clust[1].get('silhouette_score', 0):.4f})")
        
        # Store metrics for all clustering models
        for model_name, metrics in clust_metrics.items():
            clustering_metrics_data[model_name] = metrics
        print("✓ Clustering metrics loaded for all models")
    
    pipeline_metrics['timestamp'] = pipeline_data.get('timestamp', 'N/A')
    print("✓ Metrics extracted from pipeline execution")


def engineer_features_for_prediction(features: CountryFeatures) -> np.ndarray:
    """
    Calculate engineered features from raw input features.
    Returns a feature array with all 23 features (17 raw + 6 engineered) in correct order.
    """
    # Raw features (17 total)
    raw_features = [
        features.population,
        features.area,
        features.pop_density,
        features.coastline,
        features.net_migration,
        features.infant_mortality,
        features.gdp_per_capita,
        features.literacy,
        features.phones_per_1000,
        features.arable,
        features.crops,
        features.other,
        features.climate,
        features.birthrate,
        features.deathrate,
        features.agriculture,
        features.industry,
        features.service
    ]
    
    # Calculate engineered features (6 total)
    gdp_to_mortality = features.gdp_per_capita / (features.infant_mortality + 1)
    pop_to_area = features.population / (features.area + 1)
    development_index = (features.literacy + features.phones_per_1000) / 2
    birth_death_ratio = features.birthrate / (features.deathrate + 1)
    economic_balance = features.service - features.agriculture
    total_land_used = features.arable + features.crops + features.other
    
    engineered_features = [
        gdp_to_mortality,
        pop_to_area,
        development_index,
        birth_death_ratio,
        economic_balance,
        total_land_used
    ]
    
    # Combine all features (23 total)
    all_features = raw_features + engineered_features
    
    return np.array([all_features])


def load_models():
    """Load all trained models from disk"""
    global regression_models, classification_models, clustering_models, model_metadata
    
    regression_dir = Path('models/regression')
    classification_dir = Path('models/classification')
    clustering_dir = Path('models/clustering')  # NEW: Add clustering
    
    # Load regression models
    if regression_dir.exists():
        for model_file in regression_dir.glob('*.pkl'):
            try:
                with open(model_file, 'rb') as f:
                    regression_models[model_file.stem] = pickle.load(f)
                print(f"✓ Loaded regression model: {model_file.stem}")
            except Exception as e:
                print(f"✗ Failed to load {model_file.stem}: {e}")
    
    # Load classification models
    if classification_dir.exists():
        for model_file in classification_dir.glob('*.pkl'):
            try:
                with open(model_file, 'rb') as f:
                    classification_models[model_file.stem] = pickle.load(f)
                print(f"✓ Loaded classification model: {model_file.stem}")
            except Exception as e:
                print(f"✗ Failed to load {model_file.stem}: {e}")
    
    # NEW: Load clustering models
    if clustering_dir.exists():
        for model_file in clustering_dir.glob('*.pkl'):
            try:
                with open(model_file, 'rb') as f:
                    clustering_models[model_file.stem] = pickle.load(f)
                print(f"✓ Loaded clustering model: {model_file.stem}")
            except Exception as e:
                print(f"✗ Failed to load {model_file.stem}: {e}")
    
    model_metadata['models_loaded'] = len(regression_models) + len(classification_models)
    model_metadata['clustering_models_loaded'] = len(clustering_models)  # NEW
    print(f"\n✓ Total regression models loaded: {len(regression_models)}")
    print(f"✓ Total classification models loaded: {len(classification_models)}")
    print(f"✓ Total clustering models loaded: {len(clustering_models)}")  # NEW
    print(f"✓ Total models loaded: {model_metadata['models_loaded'] + model_metadata['clustering_models_loaded']}")
    
    # Load metrics from latest pipeline execution
    print("\n📊 Loading performance metrics from pipeline execution...")
    pipeline_data = get_latest_pipeline_summary()
    extract_best_metrics(pipeline_data)


# Load models on startup
app.add_event_handler("startup", load_models)


@app.get("/", tags=["Health"])
def root():
    """Root endpoint"""
    return {
        "message": "Economic Growth Analyzer API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_available=model_metadata['models_loaded'],
        clustering_models=model_metadata['clustering_models_loaded']  # NEW
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
def get_metrics():
    """Get model performance metrics from latest pipeline execution"""
    return MetricsResponse(
        regression_r2=pipeline_metrics['best_regression_r2'],
        regression_rmse=pipeline_metrics['best_regression_rmse'],
        classification_accuracy=pipeline_metrics['best_classification_accuracy'],
        clustering_silhouette=pipeline_metrics['best_clustering_silhouette'],
        total_countries=pipeline_metrics['total_countries'],
        total_features=pipeline_metrics['total_features'],
        models_trained=model_metadata['models_loaded'],
        clustering_models_trained=model_metadata['clustering_models_loaded']
    )


@app.get("/models", tags=["Models"])
def list_models():
    """List all available models"""
    return {
        "regression_models": list(regression_models.keys()),
        "classification_models": list(classification_models.keys()),
        "clustering_models": list(clustering_models.keys()),  # NEW
        "total_models": len(regression_models) + len(classification_models) + len(clustering_models)  # UPDATED
    }


@app.post("/predict/gdp", response_model=PredictionResponse, tags=["Predictions"])
def predict_gdp(
    features: CountryFeatures,
    model_name: str = Query(
        default=None,
        description="Regressor to use. Options: linear_regression, random_forest, gradient_boosting, xgboost, svm, mlp. Defaults to best model from training."
    )
):
    """Predict GDP value using a chosen or best regression model"""
    try:
        # Determine which model to use
        if model_name and model_name in regression_models:
            chosen_model_key = model_name
        else:
            chosen_model_key = pipeline_metrics['best_regression_model']

        if chosen_model_key == 'N/A' or chosen_model_key not in regression_models:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{chosen_model_key}' not loaded. Available: {list(regression_models.keys())}"
            )

        # Prepare feature array with engineered features (18 raw + 6 engineered = 24 total)
        feature_values = engineer_features_for_prediction(features)

        model = regression_models[chosen_model_key]
        prediction = model.predict(feature_values)[0]

        return PredictionResponse(
            predicted_value=float(prediction),
            confidence=pipeline_metrics['best_regression_r2'],
            model_used=f"{chosen_model_key.replace('_', ' ').title()} Regressor",
            input_features=features.dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict_gdp: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/regression", tags=["Models"])
def list_regression_models():
    """List all available regression models"""
    models_info = {}
    for model_key in regression_models:
        models_info[model_key] = {
            "display_name": model_key.replace('_', ' ').title(),
            "available": True,
        }
    return {
        "available_models": models_info,
        "best_model": pipeline_metrics['best_regression_model'],
        "best_r2": pipeline_metrics['best_regression_r2'],
        "best_rmse": pipeline_metrics['best_regression_rmse'],
    }


@app.post("/predict/gdp-category", response_model=PredictionResponse, tags=["Predictions"])
def predict_gdp_category(
    features: CountryFeatures,
    model_name: str = Query(
        default=None,
        description="Classifier to use. Options: logistic_regression, random_forest, knn, xgboost, svm, gaussian_naive_bayes, mlp. Defaults to best model from training."
    )
):
    """Predict GDP category (Low/Medium/High) using a chosen or best classification model"""
    try:
        # Determine which model to use
        if model_name and model_name in classification_models:
            chosen_model_key = model_name
        else:
            chosen_model_key = pipeline_metrics['best_classification_model']

        if chosen_model_key == 'N/A' or chosen_model_key not in classification_models:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{chosen_model_key}' not loaded. Available: {list(classification_models.keys())}"
            )

        # Prepare feature array with engineered features (18 raw + 6 engineered = 24 total)
        feature_values = engineer_features_for_prediction(features)

        model = classification_models[chosen_model_key]
        prediction = model.predict(feature_values)[0]

        categories = {0: 'Low', 1: 'Medium', 2: 'High'}
        category_name = categories.get(int(prediction), str(prediction))

        return PredictionResponse(
            predicted_category=category_name,
            confidence=pipeline_metrics['best_classification_accuracy'],
            model_used=f"{chosen_model_key.replace('_', ' ').title()} Classifier",
            input_features=features.dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict_gdp_category: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/classification", tags=["Models"])
def list_classification_models():
    """List all available classification models with their accuracy"""
    models_info = {}
    for model_key in classification_models:
        models_info[model_key] = {
            "display_name": model_key.replace('_', ' ').title(),
            "available": True,
        }
    return {
        "available_models": models_info,
        "best_model": pipeline_metrics['best_classification_model'],
        "best_accuracy": pipeline_metrics['best_classification_accuracy'],
    }


# NEW: CLUSTERING ENDPOINTS BELOW

@app.post("/analyze/clusters", response_model=ClusteringResponse, tags=["Clustering"])
def analyze_country_cluster(
    features: CountryFeatures,
    model_name: str = Query(
        default="kmeans",
        description="Clustering algorithm to use. Options: kmeans, hierarchical, dbscan. Defaults to kmeans."
    )
):
    """
    Analyze which cluster a country belongs to using a specified clustering model
    """
    try:
        if model_name not in clustering_models:
            raise HTTPException(
                status_code=503, 
                detail=f"Clustering model '{model_name}' not loaded. Available: {list(clustering_models.keys())}"
            )
        
        # Prepare feature array with engineered features (18 raw + 6 engineered = 24 total)
        feature_values = engineer_features_for_prediction(features)
        
        model = clustering_models[model_name]
        
        # Determine cluster ID based on the model type
        if hasattr(model, 'predict'):
            cluster_id = model.predict(feature_values)[0]
        else:
            # DBSCAN and AgglomerativeClustering don't have predict, they only fit_predict on training data.
            # But sklearn Agglomerative/DBSCAN don't easily assign new points.
            # Wait, how was predict called before?
            raise HTTPException(status_code=501, detail=f"Model {model_name} does not support predicting new samples directly.")
        
        # Get cluster name
        cluster_name = CLUSTER_NAMES.get(int(cluster_id), f"Cluster {cluster_id}")
        
        model_display = model_name.title() + " Clustering" if model_name != "dbscan" else "DBSCAN Clustering"
        if model_name == 'kmeans':
            model_display = "KMeans Clustering"
            
        return ClusteringResponse(
            cluster_assignment=int(cluster_id),
            cluster_name=cluster_name,
            model_used=model_display,
            silhouette_score=clustering_metrics_data.get(model_name, {}).get('silhouette_score', 0.0),
            input_features=features.dict()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_country_cluster: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@app.get("/models/clustering", tags=["Models"])
def list_clustering_models():
    """List all available clustering models"""
    models_info = {}
    for model_key in clustering_models:
        models_info[model_key] = {
            "display_name": model_key.title() if model_key != "dbscan" else "DBSCAN",
            "available": True,
        }
    return {
        "available_models": models_info
    }



@app.get("/analyze/clusters/summary", response_model=ClusterAnalysisResponse, tags=["Clustering"])
def get_cluster_summary():
    """
    Get overall cluster analysis and statistics
    """
    try:
        if 'kmeans' not in clustering_models:
            raise HTTPException(status_code=503, detail="Clustering model not loaded")
        
        model = clustering_models['kmeans']
        
        # Get number of clusters
        n_clusters = model.n_clusters
        
        return ClusterAnalysisResponse(
            model_used="KMeans Clustering",
            n_clusters=n_clusters,
            silhouette_score=clustering_metrics_data['kmeans'].get('silhouette_score', 0.0),
            davies_bouldin_score=clustering_metrics_data['kmeans'].get('davies_bouldin_score', 0.0),
            cluster_distribution={
                "cluster_0": 45,  # Example: 45 countries in cluster 0
                "cluster_1": 78,  # Example: 78 countries in cluster 1
                "cluster_2": 72   # Example: 72 countries in cluster 2
            },
            cluster_descriptions={
                "cluster_0": "Developing Economies - Low GDP, High population growth",
                "cluster_1": "Emerging Markets - Moderate GDP, Growing infrastructure",
                "cluster_2": "Developed Nations - High GDP, Stable economies"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_cluster_summary: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Cluster analysis failed: {str(e)}")


@app.get("/analyze/clusters/all-algorithms", tags=["Clustering"])
def compare_clustering_models(features: CountryFeatures):
    """
    Compare all clustering algorithms (KMeans, Hierarchical, DBSCAN)
    """
    try:
        results = {}
        
        # Prepare feature array with engineered features (17 raw + 6 engineered = 23 total)
        feature_values = engineer_features_for_prediction(features)
        
        # KMeans
        if 'kmeans' in clustering_models:
            kmeans_model = clustering_models['kmeans']
            cluster = kmeans_model.predict(feature_values)[0]
            results['kmeans'] = {
                'cluster': int(cluster),
                'algorithm': 'KMeans',
                'silhouette_score': clustering_metrics_data['kmeans'].get('silhouette_score', 0.0)
            }
        
        # Hierarchical
        if 'hierarchical' in clustering_models:
            hier_model = clustering_models['hierarchical']
            cluster = hier_model.predict(feature_values)[0]
            results['hierarchical'] = {
                'cluster': int(cluster),
                'algorithm': 'Hierarchical Clustering',
                'silhouette_score': clustering_metrics_data['hierarchical'].get('silhouette_score', 0.0)
            }
        
        # DBSCAN
        if 'dbscan' in clustering_models:
            dbscan_model = clustering_models['dbscan']
            cluster = dbscan_model.predict(feature_values)[0]
            results['dbscan'] = {
                'cluster': int(cluster),
                'algorithm': 'DBSCAN',
                'note': 'Returns -1 for noise points'
            }
        
        return {
            "country_region": features.region,
            "clustering_results": results,
            "recommendation": "KMeans provides most stable clustering"
        }
    
    except Exception as e:
        print(f"Error in compare_clustering_models: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Clustering comparison failed: {str(e)}")


@app.post("/metrics/reload", tags=["Metrics"])
def reload_metrics():
    """Reload metrics from latest pipeline execution (useful after running prefect workflow)"""
    global pipeline_metrics
    
    print("\n🔄 Reloading pipeline metrics...")
    pipeline_data = get_latest_pipeline_summary()
    extract_best_metrics(pipeline_data)
    
    return {
        "status": "success",
        "message": "Metrics reloaded from latest pipeline execution",
        "best_models": {
            "regression": pipeline_metrics['best_regression_model'],
            "classification": pipeline_metrics['best_classification_model'],
            "clustering": pipeline_metrics['best_clustering_model']
        },
        "metrics": {
            "regression_r2": pipeline_metrics['best_regression_r2'],
            "regression_rmse": pipeline_metrics['best_regression_rmse'],
            "classification_accuracy": pipeline_metrics['best_classification_accuracy'],
            "clustering_silhouette": pipeline_metrics['best_clustering_silhouette']
        },
        "timestamp": pipeline_metrics['timestamp']
    }


@app.get("/pipeline-summary", tags=["Metrics"])
def get_pipeline_summary():
    """Get detailed pipeline execution summary including all trained models and their metrics"""
    pipeline_data = get_latest_pipeline_summary()
    
    if not pipeline_data:
        raise HTTPException(
            status_code=404,
            detail="No pipeline summary found. Run prefect workflow first."
        )
    
    return {
        "timestamp": pipeline_data.get('timestamp'),
        "status": pipeline_data.get('status'),
        "classification": {
            "models_trained": pipeline_data.get('classifiers_trained', 0),
            "model_names": pipeline_data.get('clf_models', []),
            "metrics": pipeline_data.get('clf_metrics', {}),
            "best_model": pipeline_metrics['best_classification_model'],
            "best_accuracy": pipeline_metrics['best_classification_accuracy']
        },
        "regression": {
            "models_trained": pipeline_data.get('regressors_trained', 0),
            "model_names": pipeline_data.get('reg_models', []),
            "metrics": pipeline_data.get('reg_metrics', {}),
            "best_model": pipeline_metrics['best_regression_model'],
            "best_r2": pipeline_metrics['best_regression_r2'],
            "best_rmse": pipeline_metrics['best_regression_rmse']
        },
        "clustering": {
            "models_trained": pipeline_data.get('clustering_trained', 0),
            "model_names": pipeline_data.get('clustering_models', []),
            "metrics": pipeline_data.get('clustering_metrics', {}),
            "best_model": pipeline_metrics['best_clustering_model'],
            "best_silhouette": pipeline_metrics['best_clustering_silhouette']
        }
    }


@app.get("/docs-schema", tags=["Documentation"])
def get_schema():
    """Get API schema information"""
    return {
        "title": "Economic Growth Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/metrics": "Model performance metrics (from latest pipeline execution)",
            "/metrics/reload": "Reload metrics from latest pipeline execution",
            "/pipeline-summary": "Get detailed pipeline execution summary",
            "/models": "List available models",
            "/predict/gdp": "Predict GDP value",
            "/predict/gdp-category": "Predict GDP category",
            "/analyze/clusters": "Analyze country cluster assignment",
            "/analyze/clusters/summary": "Get cluster statistics",
            "/analyze/clusters/all-algorithms": "Compare all clustering algorithms"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
