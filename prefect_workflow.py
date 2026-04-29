"""
Prefect Workflow for Economic Growth Analyzer ML Pipeline

This workflow orchestrates the complete machine learning pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training (classification and regression)
- Model evaluation and persistence
- Saving execution results to file

Run with: prefect deploy prefect_workflow.py:ml_training_flow
Or locally: python -c "from prefect_workflow import ml_training_flow; ml_training_flow()"
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from prefect import flow, task
from prefect.context import get_run_context
from prefect.logging import get_run_logger

from src.models.training import TrainingPipeline


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging to file and console"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prefect_workflow_{timestamp}.log'
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file


# ============================================================================
# TASKS - Atomic units of work
# ============================================================================

@task(name="Load and Preprocess Data", retries=2)
def load_preprocess_task(data_path: str = 'countries of the world.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the dataset."""
    logger = get_run_logger()
    logger.info(f"Starting data loading from: {data_path}")
    
    pipeline = TrainingPipeline(data_path)
    df, df_processed = pipeline.load_and_preprocess()
    
    logger.info(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, df_processed


@task(name="Engineer Features")
def engineer_features_task(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the dataset."""
    logger = get_run_logger()
    logger.info("Starting feature engineering")
    
    pipeline = TrainingPipeline()
    pipeline.df = df
    df_engineered = pipeline.engineer_features()
    
    logger.info(f"✓ Features engineered. Shape: {df_engineered.shape}")
    return df_engineered


@task(name="Create Target Variable")
def create_target_task(df: pd.DataFrame) -> pd.DataFrame:
    """Create GDP category target variable for classification."""
    logger = get_run_logger()
    logger.info("Creating target variable")
    
    pipeline = TrainingPipeline()
    pipeline.df = df
    df_with_target = pipeline.create_target_variable()
    
    logger.info(f"✓ Target variable created with categories")
    return df_with_target


@task(name="Prepare Train-Test Split")
def prepare_split_task(df: pd.DataFrame) -> Dict:
    """Prepare training and test datasets."""
    logger = get_run_logger()
    logger.info("Preparing train-test split (80-20)")
    
    pipeline = TrainingPipeline()
    pipeline.df = df
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = pipeline.prepare_train_test_split()
    
    split_info = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'n_features': X_train.shape[1]
    }
    
    logger.info(f"✓ Split prepared - Train: {split_info['train_size']}, Test: {split_info['test_size']}, Features: {split_info['n_features']}")
    return split_info


@task(name="Train Classification Models", retries=1)
def train_classification_task(split_data: Dict) -> Dict:
    """Train all classification models."""
    logger = get_run_logger()
    logger.info("Starting classification model training")
    
    pipeline = TrainingPipeline()
    clf_manager = pipeline.train_classification_models_with_data(
        split_data['X_train'], split_data['y_train_clf'],
        split_data['X_test'], split_data['y_test_clf']
    )
    
    metrics = {
        'classifiers_trained': len(clf_manager.models),
        'model_names': list(clf_manager.models.keys()),
        'metrics': clf_manager.metrics
    }
    
    logger.info(f"✓ Classification complete - {metrics['classifiers_trained']} models trained")
    return metrics


@task(name="Train Regression Models", retries=1)
def train_regression_task(split_data: Dict) -> Dict:
    """Train all regression models."""
    logger = get_run_logger()
    logger.info("Starting regression model training")
    
    pipeline = TrainingPipeline()
    reg_manager = pipeline.train_regression_models_with_data(
        split_data['X_train'], split_data['y_train_reg'],
        split_data['X_test'], split_data['y_test_reg']
    )
    
    metrics = {
        'regressors_trained': len(reg_manager.models),
        'model_names': list(reg_manager.models.keys()),
        'metrics': reg_manager.metrics
    }
    
    logger.info(f"✓ Regression complete - {metrics['regressors_trained']} models trained")
    return metrics

@task(name="Train Clustering Models", retries=1)
def train_clustering_task(split_data: Dict) -> Dict:
    """Train all clustering models."""
    logger = get_run_logger()
    logger.info("Starting clustering model training")
    
    from src.models.clustering_models import ClusteringModelManager
    
    # Get features (same as classification/regression)
    X = split_data['X_train']
    
    clust_manager = ClusteringModelManager()
    clust_manager.train_all_clustering(X, n_clusters=3)
    clust_manager.save_models()
    
    metrics = {
        'clustering_models_trained': len(clust_manager.models),
        'model_names': list(clust_manager.models.keys()),
        'metrics': clust_manager.metrics
    }
    
    logger.info(f"✓ Clustering complete - {metrics['clustering_models_trained']} models trained")
    return metrics

@task(name="Validate and Log Results")
def validate_results_task(clf_metrics: Dict, reg_metrics: Dict) -> Dict:
    """Validate training results and log summary."""
    logger = get_run_logger()
    logger.info("Validating training results")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'classifiers_trained': clf_metrics['classifiers_trained'],
        'regressors_trained': reg_metrics['regressors_trained'],
        'clf_models': clf_metrics['model_names'],
        'reg_models': reg_metrics['model_names'],
        'clf_metrics': clf_metrics['metrics'],
        'reg_metrics': reg_metrics['metrics'],
        'status': 'success'
    }
    
    best_clf = max(clf_metrics['metrics'].items(), 
                   key=lambda x: x[1].get('test_accuracy', 0))
    best_reg = max(reg_metrics['metrics'].items(), 
                   key=lambda x: x[1].get('test_r2', 0))
    
    logger.info(f"✓ Best Classifier: {best_clf[0]} (Accuracy: {best_clf[1].get('test_accuracy', 0):.3f})")
    logger.info(f"✓ Best Regressor: {best_reg[0]} (R²: {best_reg[1].get('test_r2', 0):.3f})")
    
    return summary


@task(name="Save Results to File")
def save_results_task(summary: Dict, output_dir: str = 'results') -> str:
    """Save the final pipeline summary to a JSON file."""
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    # Format a safe filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = out_path / f'pipeline_summary_{timestamp}.json'
    
    # Write summary dictionary to JSON file
    with open(file_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"✓ Pipeline execution results saved to: {file_path}")
    return str(file_path)


# ============================================================================
# FLOW - Main workflow orchestration
# ============================================================================

@flow(name="ML Training Pipeline", description="Complete ML pipeline for Economic Growth Analyzer")
def ml_training_flow(
    data_path: str = 'countries of the world.csv',
    run_classification: bool = True,
    run_regression: bool = True,
    run_clustering: bool = True  
) -> Dict:
    """
    Main Prefect flow orchestrating the complete ML training pipeline.
    """
    
    logger = get_run_logger()
    context = get_run_context()
    logger.info("=" * 70)
    logger.info(f"🚀 Starting ML Training Pipeline - Run ID: {context.flow_run.id}")
    logger.info("=" * 70)
    
    # Steps 1-4: Same as before
    df, df_processed = load_preprocess_task(data_path)
    df_engineered = engineer_features_task(df)
    df_with_target = create_target_task(df_engineered)
    split_data = prepare_split_task(df_with_target)
    
    # Step 5, 6, 7: Train all three model types
    clf_results = None
    reg_results = None
    clust_results = None
    
    if run_classification:
        clf_results = train_classification_task(split_data)
    
    if run_regression:
        reg_results = train_regression_task(split_data)
    
    if run_clustering:  # ADD THIS
        clust_results = train_clustering_task(split_data)
    
    # Step 8: Validate and Extract Summary
    if clf_results and reg_results and clust_results:  # UPDATED
        summary = validate_results_task(clf_results, reg_results)
        # Add clustering to summary
        summary['clustering_trained'] = clust_results['clustering_models_trained']
        summary['clustering_models'] = clust_results['model_names']
        summary['clustering_metrics'] = clust_results['metrics']
    else:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'status': 'partial',
            'classification_run': run_classification,
            'regression_run': run_regression,
            'clustering_run': run_clustering
        }
    
    # Step 9: Save results to disk
    saved_file_path = save_results_task(summary)
    summary['saved_file_path'] = saved_file_path
    
    logger.info("=" * 70)
    logger.info("✅ ML Training Pipeline Completed Successfully!")
    logger.info("=" * 70)
    
    return summary


# ============================================================================
# ALTERNATIVE: Scheduled Flow (runs on a schedule)
# ============================================================================

@flow(name="Scheduled ML Training", description="ML pipeline scheduled to run periodically")
def scheduled_ml_training_flow() -> Dict:
    """
    Scheduled version of ML training flow for production deployments.
    """
    return ml_training_flow(
        data_path='countries of the world.csv',
        run_classification=True,
        run_regression=True
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    log_file = setup_logging()
    local_logger = logging.getLogger(__name__)
    local_logger.info(f"Running ML Training Pipeline locally... Logs: {log_file}")

    result = ml_training_flow(
        data_path='countries of the world.csv',
        run_classification=True,
        run_regression=True,
        run_clustering=True  
    )

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Status: {result.get('status', 'N/A')}")
    print(f"Classifiers Trained: {result.get('classifiers_trained', 0)}")
    print(f"Regressors Trained: {result.get('regressors_trained', 0)}")
    print(f"Clustering Models Trained: {result.get('clustering_trained', 0)}")  # ADD THIS
    print(f"Timestamp: {result.get('timestamp', 'N/A')}")
    print(f"Results File: {result.get('saved_file_path', 'N/A')}")
    print("=" * 70)