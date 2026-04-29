"""
Prefect Workflow for Economic Growth Analyzer ML Pipeline

This workflow orchestrates the complete machine learning pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training (classification and regression)
- Model evaluation and persistence

Run with: prefect deploy prefect_workflow.py:ml_training_flow
Or locally: python -c "from prefect_workflow import ml_training_flow; ml_training_flow()"
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from prefect import flow, task, logger
from prefect.context import get_run_context

from src.models.training import TrainingPipeline


# ============================================================================
# TASKS - Atomic units of work
# ============================================================================

@task(name="Load and Preprocess Data", retries=2)
def load_preprocess_task(data_path: str = 'countries of the world.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the dataset.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Tuple of (processed_df, original_df)
    """
    logger.info(f"Starting data loading from: {data_path}")
    
    pipeline = TrainingPipeline(data_path)
    df, df_processed = pipeline.load_and_preprocess()
    
    logger.info(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, df_processed


@task(name="Engineer Features")
def engineer_features_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with engineered features
    """
    logger.info("Starting feature engineering")
    
    pipeline = TrainingPipeline()
    pipeline.df = df
    df_engineered = pipeline.engineer_features()
    
    logger.info(f"✓ Features engineered. Shape: {df_engineered.shape}")
    return df_engineered


@task(name="Create Target Variable")
def create_target_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create GDP category target variable for classification.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with GDP_Category column added
    """
    logger.info("Creating target variable")
    
    pipeline = TrainingPipeline()
    pipeline.df = df
    df_with_target = pipeline.create_target_variable()
    
    logger.info(f"✓ Target variable created with categories")
    return df_with_target


@task(name="Prepare Train-Test Split")
def prepare_split_task(df: pd.DataFrame) -> Dict:
    """
    Prepare training and test datasets.
    
    Args:
        df: Input dataframe with all features and targets
        
    Returns:
        Dictionary with X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf
    """
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
    """
    Train all classification models.
    
    Args:
        split_data: Dictionary with train/test splits
        
    Returns:
        Dictionary with metrics for each classifier
    """
    logger.info("Starting classification model training")
    
    pipeline = TrainingPipeline()
    clf_manager = pipeline.train_classification_models_with_data(
        split_data['X_train'], split_data['y_train_clf'],
        split_data['X_test'], split_data['y_test_clf']
    )
    
    # Extract metrics
    metrics = {
        'classifiers_trained': len(clf_manager.models),
        'model_names': list(clf_manager.models.keys()),
        'metrics': clf_manager.metrics
    }
    
    logger.info(f"✓ Classification complete - {metrics['classifiers_trained']} models trained")
    return metrics


@task(name="Train Regression Models", retries=1)
def train_regression_task(split_data: Dict) -> Dict:
    """
    Train all regression models.
    
    Args:
        split_data: Dictionary with train/test splits
        
    Returns:
        Dictionary with metrics for each regressor
    """
    logger.info("Starting regression model training")
    
    pipeline = TrainingPipeline()
    reg_manager = pipeline.train_regression_models_with_data(
        split_data['X_train'], split_data['y_train_reg'],
        split_data['X_test'], split_data['y_test_reg']
    )
    
    # Extract metrics
    metrics = {
        'regressors_trained': len(reg_manager.models),
        'model_names': list(reg_manager.models.keys()),
        'metrics': reg_manager.metrics
    }
    
    logger.info(f"✓ Regression complete - {metrics['regressors_trained']} models trained")
    return metrics


@task(name="Validate and Log Results")
def validate_results_task(clf_metrics: Dict, reg_metrics: Dict) -> Dict:
    """
    Validate training results and log summary.
    
    Args:
        clf_metrics: Classification metrics
        reg_metrics: Regression metrics
        
    Returns:
        Summary dictionary with validation results
    """
    logger.info("Validating training results")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'classifiers_trained': clf_metrics['classifiers_trained'],
        'regressors_trained': reg_metrics['regressors_trained'],
        'clf_models': clf_metrics['model_names'],
        'reg_models': reg_metrics['model_names'],
        'status': 'success'
    }
    
    # Log best performers
    best_clf = max(clf_metrics['metrics'].items(), 
                   key=lambda x: x[1].get('test_accuracy', 0))
    best_reg = max(reg_metrics['metrics'].items(), 
                   key=lambda x: x[1].get('test_r2', 0))
    
    logger.info(f"✓ Best Classifier: {best_clf[0]} (Accuracy: {best_clf[1].get('test_accuracy', 0):.3f})")
    logger.info(f"✓ Best Regressor: {best_reg[0]} (R²: {best_reg[1].get('test_r2', 0):.3f})")
    
    return summary


# ============================================================================
# FLOW - Main workflow orchestration
# ============================================================================

@flow(name="ML Training Pipeline", description="Complete ML pipeline for Economic Growth Analyzer")
def ml_training_flow(
    data_path: str = 'countries of the world.csv',
    run_classification: bool = True,
    run_regression: bool = True
) -> Dict:
    """
    Main Prefect flow orchestrating the complete ML training pipeline.
    
    This flow:
    1. Loads and preprocesses data
    2. Engineers features
    3. Creates target variables
    4. Splits data into train/test sets
    5. Trains classification models
    6. Trains regression models
    7. Validates and logs results
    
    Args:
        data_path: Path to the dataset
        run_classification: Whether to train classification models
        run_regression: Whether to train regression models
        
    Returns:
        Summary dictionary with pipeline results
    """
    
    context = get_run_context()
    logger.info("=" * 70)
    logger.info(f"🚀 Starting ML Training Pipeline - Run ID: {context.flow_run.id}")
    logger.info("=" * 70)
    
    # Step 1: Load and Preprocess
    df, df_processed = load_preprocess_task(data_path)
    
    # Step 2: Feature Engineering
    df_engineered = engineer_features_task(df)
    
    # Step 3: Create Target Variable
    df_with_target = create_target_task(df_engineered)
    
    # Step 4: Prepare Train-Test Split
    split_data = prepare_split_task(df_with_target)
    
    # Step 5 & 6: Train Models (can run in parallel)
    clf_results = None
    reg_results = None
    
    if run_classification:
        clf_results = train_classification_task(split_data)
    
    if run_regression:
        reg_results = train_regression_task(split_data)
    
    # Step 7: Validate and Log Results
    if clf_results and reg_results:
        summary = validate_results_task(clf_results, reg_results)
    else:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'status': 'partial',
            'classification_run': run_classification,
            'regression_run': run_regression
        }
    
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
    Can be scheduled to run daily, weekly, etc.
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
    # Local execution
    logger.info("Running ML Training Pipeline locally...")
    
    result = ml_training_flow(
        data_path='countries of the world.csv',
        run_classification=True,
        run_regression=True
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Status: {result.get('status', 'N/A')}")
    print(f"Classifiers Trained: {result.get('classifiers_trained', 0)}")
    print(f"Regressors Trained: {result.get('regressors_trained', 0)}")
    print(f"Timestamp: {result.get('timestamp', 'N/A')}")
    print("=" * 70)
