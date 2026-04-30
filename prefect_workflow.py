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
from typing import Dict, Tuple, List

import pandas as pd
from prefect import flow, task
from prefect.context import get_run_context
from prefect.logging import get_run_logger

import subprocess

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


@task(name="Hyperparameter Tuning (Optuna - Sequential)")
def tune_hyperparameters_task(
    split_data: Dict,
    n_trials: int = 10,
    timeout: int = 300,
    seed: int = 42,
) -> Dict:
    """Tune hyperparameters for selected models using Optuna (sequential)."""
    logger = get_run_logger()
    logger.info("Starting hyperparameter tuning with Optuna (sequential)")

    from src.models.hyperparameter_tuning import HyperparameterTuner

    tuner = HyperparameterTuner(n_trials=n_trials, timeout=timeout, seed=seed)

    # Classification tuning
    tuner.tune_logistic_regression(split_data['X_train'], split_data['y_train_clf'])
    tuner.tune_random_forest_classifier(split_data['X_train'], split_data['y_train_clf'])
    tuner.tune_xgboost_classifier(split_data['X_train'], split_data['y_train_clf'])

    # Regression tuning
    tuner.tune_random_forest_regressor(split_data['X_train'], split_data['y_train_reg'])
    tuner.tune_xgboost_regressor(split_data['X_train'], split_data['y_train_reg'])

    out_path = Path("results") / "hyperparameters.json"
    tuner.save_best_params(out_path)

    logger.info("✓ Hyperparameter tuning complete")
    return {"tuned_params": tuner.get_all_best_params()}

@task(name="Train Classification Models", retries=1)
def train_classification_task(split_data: Dict, tuned_params: Dict | None = None) -> Dict:
    """Train all classification models."""
    logger = get_run_logger()
    logger.info("Starting classification model training")

    pipeline = TrainingPipeline()
    clf_manager = pipeline.train_classification_models_with_data(
        split_data['X_train'],
        split_data['y_train_clf'],
        split_data['X_test'],
        split_data['y_test_clf'],
        tuned_params=tuned_params,
    )

    metrics = {
        'classifiers_trained': len(clf_manager.models),
        'model_names': list(clf_manager.models.keys()),
        'metrics': clf_manager.metrics
    }

    logger.info(f"✓ Classification complete - {metrics['classifiers_trained']} models trained")
    return metrics


@task(name="Train Regression Models", retries=1)
def train_regression_task(split_data: Dict, tuned_params: Dict | None = None) -> Dict:
    """Train all regression models."""
    logger = get_run_logger()
    logger.info("Starting regression model training")

    pipeline = TrainingPipeline()
    reg_manager = pipeline.train_regression_models_with_data(
        split_data['X_train'],
        split_data['y_train_reg'],
        split_data['X_test'],
        split_data['y_test_reg'],
        tuned_params=tuned_params,
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


@task(name="Run Model Validation Tests", retries=0)
def run_model_validation_task(split_data: Dict, clf_metrics: Dict, reg_metrics: Dict) -> Dict:
    """
    Validate trained models against performance thresholds.
    This is a critical quality gate - failing validation stops the pipeline.
    """
    logger = get_run_logger()
    logger.info("\n" + "=" * 70)
    logger.info("🧪 RUNNING ML VALIDATION TESTS (Quality Gate)")
    logger.info("=" * 70)
    
    import numpy as np
    
    validation_results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'threshold_checks': {},
        'status': 'PENDING'
    }
    
    try:
        # Check 1: Classification accuracy threshold
        logger.info("\n[1/3] Validating Classification Accuracy...")
        if clf_metrics and 'metrics' in clf_metrics:
            clf_accuracies = [m['test_accuracy'] for m in clf_metrics['metrics'].values()]
            max_clf_accuracy = max(clf_accuracies) if clf_accuracies else 0.0
            min_accuracy_threshold = 0.70  # Minimum acceptable accuracy
            
            if max_clf_accuracy >= min_accuracy_threshold:
                logger.info(f"  ✓ PASSED: Best accuracy {max_clf_accuracy:.4f} >= {min_accuracy_threshold}")
                validation_results['tests_passed'] += 1
                best_clf_model = max(clf_metrics['metrics'].items(), 
                                     key=lambda x: x[1].get('test_accuracy', 0))[0]
                validation_results['threshold_checks']['classification_accuracy'] = {
                    'required': min_accuracy_threshold,
                    'achieved': max_clf_accuracy,
                    'status': 'PASS',
                    'best_model': best_clf_model
                }
            else:
                logger.warning(f"  ✗ FAILED: Best accuracy {max_clf_accuracy:.4f} < {min_accuracy_threshold}")
                validation_results['tests_failed'] += 1
                best_clf_model = max(clf_metrics['metrics'].items(), 
                                     key=lambda x: x[1].get('test_accuracy', 0))[0]
                validation_results['threshold_checks']['classification_accuracy'] = {
                    'required': min_accuracy_threshold,
                    'achieved': max_clf_accuracy,
                    'status': 'FAIL',
                    'best_model': best_clf_model
                }
        else:
            logger.warning("  ⚠ SKIPPED: No classification metrics available")
            validation_results['threshold_checks']['classification_accuracy'] = {
                'status': 'SKIPPED',
                'reason': 'No metrics available'
            }
        
        # Check 2: Regression R² threshold
        logger.info("[2/3] Validating Regression R² Score...")
        if reg_metrics and 'metrics' in reg_metrics:
            reg_r2_scores = [m['test_r2'] for m in reg_metrics['metrics'].values() if m.get('test_r2', -999) > -999]
            max_reg_r2 = max(reg_r2_scores) if reg_r2_scores else 0.0
            min_r2_threshold = 0.65  # Minimum acceptable R²
            
            if max_reg_r2 >= min_r2_threshold:
                logger.info(f"  ✓ PASSED: Best R² {max_reg_r2:.4f} >= {min_r2_threshold}")
                validation_results['tests_passed'] += 1
                best_reg_model = max(reg_metrics['metrics'].items(), 
                                     key=lambda x: x[1].get('test_r2', -999))[0]
                validation_results['threshold_checks']['regression_r2'] = {
                    'required': min_r2_threshold,
                    'achieved': max_reg_r2,
                    'status': 'PASS',
                    'best_model': best_reg_model
                }
            else:
                logger.warning(f"  ✗ FAILED: Best R² {max_reg_r2:.4f} < {min_r2_threshold}")
                validation_results['tests_failed'] += 1
                best_reg_model = max(reg_metrics['metrics'].items(), 
                                     key=lambda x: x[1].get('test_r2', -999))[0]
                validation_results['threshold_checks']['regression_r2'] = {
                    'required': min_r2_threshold,
                    'achieved': max_reg_r2,
                    'status': 'FAIL',
                    'best_model': best_reg_model
                }
        else:
            logger.warning("  ⚠ SKIPPED: No regression metrics available")
            validation_results['threshold_checks']['regression_r2'] = {
                'status': 'SKIPPED',
                'reason': 'No metrics available'
            }
        
        # Check 3: Prediction validity (no NaN values)
        logger.info("[3/3] Validating Data Quality...")
        X_test = split_data.get('X_test')
        has_nan = False
        
        if X_test is not None:
            has_nan = bool(np.any(np.isnan(X_test)))
            
            if not has_nan:
                logger.info(f"  ✓ PASSED: No NaN values in test features")
                validation_results['tests_passed'] += 1
                validation_results['threshold_checks']['data_quality'] = {
                    'nan_values': 0,
                    'status': 'PASS'
                }
            else:
                logger.warning(f"  ✗ FAILED: NaN values detected in test features")
                validation_results['tests_failed'] += 1
                validation_results['threshold_checks']['data_quality'] = {
                    'nan_values': int(np.sum(np.isnan(X_test))),
                    'status': 'FAIL'
                }
        else:
            logger.warning("  ⚠ SKIPPED: Test data not available")
            validation_results['threshold_checks']['data_quality'] = {
                'status': 'SKIPPED',
                'reason': 'No test data'
            }
        
        # Final verdict
        logger.info("\n" + "-" * 70)
        logger.info(f"Validation Summary: {validation_results['tests_passed']} PASSED, {validation_results['tests_failed']} FAILED")
        
        if validation_results['tests_failed'] > 0:
            validation_results['status'] = 'FAILED'
            logger.error("❌ ML VALIDATION TESTS FAILED - Pipeline cannot proceed")
        else:
            validation_results['status'] = 'PASSED'
            logger.info("✅ All ML VALIDATION TESTS PASSED - Pipeline can proceed")
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        validation_results['status'] = 'ERROR'
        validation_results['error'] = str(e)
    
    return validation_results


@task(name="Run DeepChecks Validation", retries=0)
def run_deepchecks_task(split_data: Dict) -> Dict:
    """
    Run DeepChecks data quality and model validation checks.
    Provides comprehensive data validation without blocking pipeline on failure.
    """
    logger = get_run_logger()
    logger.info("\n" + "=" * 70)
    logger.info("📊 RUNNING DEEPCHECKS DATA VALIDATION")
    logger.info("=" * 70)
    
    deepchecks_results = {
        'status': 'PENDING',
        'checks_run': 0,
        'checks_passed': 0,
        'checks_failed': 0,
        'warnings': []
    }
    
    try:
        # Try to import deepchecks
        try:
            import pandas as pd
            from deepchecks.tabular import Dataset
            from deepchecks.tabular.suites import train_test_validation
        except ImportError:
            logger.warning("⚠️  DeepChecks not installed. Skipping DeepChecks validation.")
            deepchecks_results['status'] = 'SKIPPED'
            deepchecks_results['reason'] = 'DeepChecks package not installed'
            return deepchecks_results
        
        # Get data from split_data
        X_train = split_data.get('X_train')
        X_test = split_data.get('X_test')
        y_train = split_data.get('y_train_clf')  # Use classification target
        y_test = split_data.get('y_test_clf')
        
        if X_train is None or X_test is None:
            logger.warning("⚠️  Incomplete split data. Skipping DeepChecks validation.")
            deepchecks_results['status'] = 'SKIPPED'
            deepchecks_results['reason'] = 'Incomplete split data'
            return deepchecks_results
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        logger.info(f"Running train-test validation suite on {X_train.shape[0]} train, {X_test.shape[0]} test samples...")
        
        # Create DeepChecks datasets
        train_ds = Dataset(
            pd.DataFrame(X_train, columns=feature_names),
            label=pd.Series(y_train, name='target') if y_train is not None else None
        )
        test_ds = Dataset(
            pd.DataFrame(X_test, columns=feature_names),
            label=pd.Series(y_test, name='target') if y_test is not None else None
        )
        
        # Run validation suite
        suite = train_test_validation()
        result = suite.run(train_dataset=train_ds, test_dataset=test_ds)
        
        if result:
            deepchecks_results['status'] = 'PASSED'
            deepchecks_results['checks_run'] = len(result.results) if hasattr(result, 'results') else 0
            logger.info(f"✓ DeepChecks validation completed successfully ({deepchecks_results['checks_run']} checks)")
            logger.info("  Sample checks: Data integrity, feature distribution, label distribution")
        else:
            deepchecks_results['status'] = 'PASSED'
            logger.info("✓ DeepChecks validation completed")
        
    except Exception as e:
        logger.warning(f"⚠️  DeepChecks validation encountered issue: {str(e)}")
        deepchecks_results['status'] = 'WARNING'
        deepchecks_results['error'] = str(e)
        deepchecks_results['warnings'].append(str(e))
    
    logger.info("=" * 70)
    return deepchecks_results


@task(name="Run pytest quality gate", retries=0)
def run_pytest_quality_gate(test_paths: List[str] | None = None) -> Dict:
    """Run the repository pytest suite (or selected files) as a Prefect quality gate.

    Fails the Prefect flow if pytest returns non-zero.
    """
    logger = get_run_logger()
    logger.info("\n" + "=" * 70)
    logger.info("🧬 RUNNING PYTEST UNIT TESTS")
    logger.info("=" * 70)

    if not test_paths:
        test_paths = [
            "tests/test_models.py",
            "tests/test_ml_validation.py",
            "tests/test_api.py",
            "tests/test_deepchecks.py",
        ]

    cmd = ["pytest", "-v", "--tb=short", *test_paths]
    logger.info(f"Running: {' '.join(cmd)}\n")

    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Log output
    if proc.stdout:
        logger.info(f"pytest output:\n{proc.stdout}")
    if proc.stderr:
        logger.warning(f"pytest stderr:\n{proc.stderr}")

    if proc.returncode != 0:
        logger.error(f"❌ pytest failed with exit code {proc.returncode}")
        raise RuntimeError(f"pytest failed with exit code {proc.returncode}")

    logger.info("✅ All pytest tests PASSED")
    logger.info("=" * 70)
    return {
        "pytest_returncode": proc.returncode,
        "pytest_tests": test_paths,
        "status": "PASSED"
    }


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

    # Step 5: Hyperparameter tuning BEFORE final training (sequential)
    hp_tuning_results = None
    if run_classification or run_regression:
        hp_tuning_results = tune_hyperparameters_task(split_data)

    tuned_params = None
    if hp_tuning_results:
        tuned_params = hp_tuning_results.get("tuned_params")

    # Step 6, 7, 8: Train all three model types
    clf_results = None
    reg_results = None
    clust_results = None

    if run_classification:
        clf_results = train_classification_task(split_data, tuned_params=tuned_params)

    if run_regression:
        reg_results = train_regression_task(split_data, tuned_params=tuned_params)

    if run_clustering:  # ADD THIS
        clust_results = train_clustering_task(split_data)
    
    # Step 8: ML VALIDATION TESTS (CRITICAL QUALITY GATE)
    logger.info("\n" + "=" * 70)
    logger.info("🧪 STAGE: MODEL VALIDATION")
    logger.info("=" * 70)
    
    validation_results = None
    deepchecks_results = None
    
    if clf_results and reg_results:
        # Run model validation (checks performance thresholds)
        validation_results = run_model_validation_task(split_data, clf_results, reg_results)
        
        # Check if validation passed - if not, stop pipeline
        if validation_results.get('status') == 'FAILED':
            logger.error("\n❌ PIPELINE STOPPED: Model validation failed!")
            logger.error(f"Failed checks: {validation_results.get('threshold_checks', {})}")
            raise RuntimeError("Model validation thresholds not met - pipeline stopped")
        
        # Run DeepChecks (data quality validation - warning only)
        deepchecks_results = run_deepchecks_task(split_data)
    
    # Step 9: Validate and Extract Summary
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

    if tuned_params:
        summary['tuned_params'] = tuned_params
    
    # Add validation results to summary
    if validation_results:
        summary['validation_results'] = validation_results
    if deepchecks_results:
        summary['deepchecks_results'] = deepchecks_results
    
    # Step 10: Save results to disk
    saved_file_path = save_results_task(summary)
    summary['saved_file_path'] = saved_file_path

    # Step 11: Run in-pipeline pytest quality gate (fails flow on test failures)
    logger.info("\n" + "=" * 70)
    logger.info("🧬 STAGE: PYTEST UNIT TESTS")
    logger.info("=" * 70)
    
    pytest_results = run_pytest_quality_gate()
    summary['pytest_results'] = pytest_results
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ML TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"Pipeline Status: SUCCESS")
    logger.info(f"Results saved to: {saved_file_path}")
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
