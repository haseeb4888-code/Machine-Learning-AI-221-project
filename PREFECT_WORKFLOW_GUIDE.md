# Prefect Workflow Guide

## Overview

The `prefect_workflow.py` file contains a Prefect workflow that orchestrates the complete ML pipeline for the Economic Growth Analyzer project. It automates:

- Data loading and preprocessing
- Feature engineering
- Target variable creation
- Train-test splitting
- Classification model training (5 models)
- Regression model training (5 models)
- Result validation and logging

## Installation

Prefect is included in the project requirements. Make sure to install it:

```bash
pip install -r requirements.txt
# or if using a virtual environment
source uenv/bin/activate
pip install -r requirements.txt
```

## Running the Workflow

### Option 1: Local Execution (Simplest)

Run the workflow locally without Prefect server:

```bash
python prefect_workflow.py
```

This will:
- Execute the complete ML pipeline sequentially
- Print progress and logging information
- Display a summary at the end

### Option 2: Prefect Cloud/Server Execution

#### Step 1: Initialize Prefect (optional)

```bash
prefect config set PREFECT_API_URL="http://localhost:4200/api"
```

#### Step 2: Start Prefect Server (in separate terminal)

```bash
prefect server start
```

#### Step 3: Register/Deploy the Flow

```bash
prefect deploy prefect_workflow.py:ml_training_flow --name "ml-training"
```

#### Step 4: Create a Run

```bash
prefect deployment run "ml_training_flow/ml-training"
```

### Option 3: Python Script Import

```python
from prefect_workflow import ml_training_flow

# Run with default parameters
result = ml_training_flow()

# Run with custom parameters
result = ml_training_flow(
    data_path='countries of the world.csv',
    run_classification=True,
    run_regression=True
)

print(f"Pipeline Status: {result['status']}")
print(f"Models Trained: {result['classifiers_trained']} classifiers, {result['regressors_trained']} regressors")
```

### Option 4: Scheduled Execution

```python
from prefect_workflow import scheduled_ml_training_flow
from prefect.schedules import cron

# Run daily at 2 AM
schedule = cron("0 2 * * *")
scheduled_ml_training_flow.serve(schedule=schedule)
```

## Workflow Structure

### Tasks (Atomic Units)

1. **load_preprocess_task** - Load and clean data
2. **engineer_features_task** - Create engineered features
3. **create_target_task** - Create GDP categories for classification
4. **prepare_split_task** - Split into train/test sets (80-20)
5. **train_classification_task** - Train 5 classifiers
6. **train_regression_task** - Train 5 regressors
7. **validate_results_task** - Validate and log results

### Main Flow

```
Load Data
    ↓
Engineer Features
    ↓
Create Target Variable
    ↓
Train-Test Split
    ↓
Train Classifiers (parallel) ← Train Regressors (parallel)
    ↓
Validate Results
    ↓
Complete
```

## Models Trained

### Classification (5 models)
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- XGBoost
- Support Vector Machine (SVM)

### Regression (5 models)
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine (SVM)

## Output

Models are automatically saved to:
- `models/classification/` - All classifier models as `.pkl` files
- `models/regression/` - All regressor models as `.pkl` files

Logs are printed to stdout and (if Prefect server is running) tracked in the Prefect UI.

## Configuration

You can customize the workflow by passing parameters:

```python
ml_training_flow(
    data_path='path/to/your/data.csv',      # Path to dataset
    run_classification=True,                 # Train classifiers
    run_regression=True                      # Train regressors
)
```

## Troubleshooting

### Prefect Server Won't Start
```bash
prefect profile create --name local
prefect server start
```

### Import Errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Verify Prefect installation
python -c "from prefect import flow; print('Prefect installed correctly')"
```

### Data File Not Found
Ensure `countries of the world.csv` is in the project root directory or provide the correct path:

```python
ml_training_flow(data_path='path/to/your/data.csv')
```

## Integration with CI/CD

For GitHub Actions or other CI/CD pipelines, use the local execution option:

```bash
python prefect_workflow.py
```

This doesn't require Prefect Cloud and will run completely offline.

## Monitoring

When using Prefect Server, you can:
- View flow run history
- Monitor task execution times
- Track logs and errors
- Retry failed tasks
- Set up alerts

Access the UI at `http://localhost:4200` when the server is running.

## Advanced Usage

### Running with Logging Configuration

```python
import logging
from prefect_workflow import ml_training_flow

logging.basicConfig(level=logging.DEBUG)
result = ml_training_flow()
```

### Running Specific Models Only

```python
# Only train classifiers
ml_training_flow(run_classification=True, run_regression=False)

# Only train regressors
ml_training_flow(run_classification=False, run_regression=True)
```

## Documentation

For more information on Prefect, visit: https://docs.prefect.io/
