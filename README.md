---
title: Machine Learning AI-221 Project1
sdk: docker
app_port: 8000
---

# Machine-Learning-AI-221-project

Economic Growth Analyzer is a machine learning project that predicts GDP-related outcomes, classifies countries by development level, and analyzes country-level patterns using a FastAPI backend, a browser frontend, and a training pipeline that produces regression, classification, and clustering models.

The repository is designed to support three major workflows:

1. Local development and experimentation
2. Automated training and validation through the Prefect workflow
3. Containerized deployment to platforms such as Hugging Face Spaces and Heroku-style container hosting

This README is intentionally detailed so it can serve as the primary project manual for setup, usage, deployment, testing, and troubleshooting.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [How the Application Works](#how-the-application-works)
- [Data and Feature Engineering](#data-and-feature-engineering)
- [Modeling Approach](#modeling-approach)
- [API Overview](#api-overview)
- [Frontend Overview](#frontend-overview)
- [Local Setup](#local-setup)
- [Running with Docker](#running-with-docker)
- [Prefect Workflow](#prefect-workflow)
- [Testing and Validation](#testing-and-validation)
- [Deployment Guide](#deployment-guide)
- [Artifacts and Outputs](#artifacts-and-outputs)
- [Troubleshooting](#troubleshooting)
- [Maintenance Notes](#maintenance-notes)
- [Credits](#credits)

## Project Overview

The goal of this project is to analyze country-level socioeconomic indicators and use machine learning to infer economic outcomes. The system combines:

- A predictive API built with FastAPI
- A browser-based frontend for interactive exploration
- Training code for multiple regression, classification, and clustering models
- A workflow layer that orchestrates preprocessing, training, validation, and reporting
- JSON summaries and model artifacts for reproducible results

The application is structured around a single dataset, [countries of the world.csv](countries%20of%20the%20world.csv), and a standard ML pipeline that transforms raw country attributes into engineered features suitable for prediction.

The deployed application exposes a user interface at the root path `/` and a metadata API at `/api`. The API also includes dedicated endpoints for health checks, model listings, metrics, and predictions.

## Key Features

### Machine Learning

- GDP prediction with regression models
- Country classification into income or development groups
- Clustering analysis for country segmentation
- Feature engineering from raw socioeconomic indicators
- Automated model selection based on the latest pipeline summary

### Web Application

- FastAPI backend
- Static frontend with a custom visual design
- Health, metrics, and model endpoints for diagnostics
- Browser-accessible documentation via `/docs`

### Workflow and Validation

- Prefect orchestration for the end-to-end pipeline
- Saved pipeline summaries in `results/`
- Automated unit tests
- Data quality checks through DeepChecks integration
- Performance gates that help prevent low-quality model releases

### Deployment

- Docker-based runtime configuration
- Hugging Face Spaces compatibility
- Container-based Heroku deployment support
- Dynamic port binding for hosted environments

## Repository Structure

The project is organized as follows:

- `src/` contains the backend and ML source code
- `frontend/` contains the single-page browser UI
- `scripts/` contains utility scripts such as model baking helpers
- `results/` stores pipeline summaries and training outputs
- `tests/` contains endpoint, data, model, validation, and integration tests
- `docs/` contains supporting documentation
- `Old-Version/` contains older notebook-based artifacts and historical code

Important files include:

- [src/api/main.py](src/api/main.py) for the FastAPI application
- [src/api/schemas.py](src/api/schemas.py) for request and response models
- [src/data/loader.py](src/data/loader.py) for dataset loading
- [src/data/preprocessor.py](src/data/preprocessor.py) for preprocessing
- [src/data/feature_engineer.py](src/data/feature_engineer.py) for feature generation
- [src/models/training.py](src/models/training.py) for training orchestration
- [src/models/classification_models.py](src/models/classification_models.py) for classification estimators
- [src/models/regression_models.py](src/models/regression_models.py) for regression estimators
- [src/models/clustering_models.py](src/models/clustering_models.py) for clustering estimators
- [prefect_workflow.py](prefect_workflow.py) for the workflow entry point
- [frontend/index.html](frontend/index.html) for the UI shell
- [frontend/app.js](frontend/app.js) for client-side behavior
- [frontend/styles.css](frontend/styles.css) for presentation

## How the Application Works

The application follows a predictable flow from raw data to deployed inference.

### 1. Data loading

The dataset is loaded from the project root. The canonical file is [countries of the world.csv](countries%20of%20the%20world.csv). Some historical and fallback copies also exist under `Old-Version/`.

### 2. Data cleaning and preprocessing

The preprocessing layer standardizes data types, handles missing values, and prepares the country-level records for modeling.

### 3. Feature engineering

The project expands the raw indicator set into engineered features that capture relationships such as:

- GDP per capita relative to infant mortality
- Population density style signals
- Development proxies based on literacy and telephones per 1000 people
- Ratios that compare birth and death rates
- Economic balance indicators from agriculture and service sectors

### 4. Train-test split

The pipeline splits the processed data into training and test sets so model performance can be measured on held-out records.

### 5. Training

The training system fits multiple candidate models across regression, classification, and clustering tasks.

### 6. Evaluation and selection

The pipeline records metrics for each trained model and stores the best-performing result for the API to use.

### 7. Deployment and inference

The API loads trained artifacts and exposes them through endpoints. If no pickled models are present, the application can still serve the frontend and metrics derived from the latest pipeline summary.

## Data and Feature Engineering

The dataset represents countries with a broad set of socioeconomic and geographic indicators. The project uses both raw and engineered features.

### Raw indicators

The prediction layer currently expects values such as:

- Population
- Area
- Population density
- Coastline
- Net migration
- Infant mortality
- GDP per capita
- Literacy
- Phones per 1000
- Arable land
- Crops
- Other land use
- Climate
- Birth rate
- Death rate
- Agriculture
- Industry
- Service
- Region

### Engineered features

The API and workflow derive additional signals from the raw fields. Examples include:

- GDP to mortality ratio
- Population to area ratio
- Development index from literacy and phones per 1000
- Birth/death ratio
- Economic balance between service and agriculture
- Total land use

These engineered features are created in [src/api/main.py](src/api/main.py) for prediction requests and in the training pipeline for model fitting.

### Why engineering matters

The engineered values make the models more expressive by encoding relationships that a raw tree or linear model might not easily infer from the base inputs alone. In practice, this can improve both accuracy and robustness.

## Modeling Approach

The project trains and compares several model families.

### Classification models

The classification stack includes approaches such as:

- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes
- Multi-Layer Perceptron
- XGBoost Classifier

The exact selection used by the workflow may vary depending on the pipeline version and configuration, but the codebase is built to support multiple classifiers.

### Regression models

The regression stack includes:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Machine Regressor
- Multi-Layer Perceptron Regressor
- XGBoost Regressor

These models are scored primarily by test-set R2 and RMSE.

### Clustering models

The clustering layer supports country segmentation through algorithms such as:

- K-Means
- Hierarchical clustering
- DBSCAN

The best clustering result is tracked through silhouette score and stored in the latest pipeline summary.

### Model persistence

Trained models are written to:

- `models/regression/`
- `models/classification/`
- `models/clustering/`

These folders are kept in the repository structure so container builds can copy them cleanly even when they contain only placeholder files.

### Loading behavior at runtime

On startup, the API scans the model directories and loads any available `.pkl` files. It then reads the latest summary JSON from `results/` to populate performance metrics.

If no model files exist, the application still starts and can still report metrics from the summary file.

## API Overview

The FastAPI application lives in [src/api/main.py](src/api/main.py).

### Main routes

- `/` serves the frontend HTML page
- `/api` returns metadata about the API
- `/health` reports service status and model counts
- `/metrics` returns the latest performance summary
- `/models` lists available model artifacts
- `/models/regression` lists regression models and scores
- `/predict/gdp` performs GDP prediction

### Important deployment behavior

The root path `/` is configured to return [frontend/index.html](frontend/index.html), which is why the hosted app shows the UI instead of raw JSON.

Static assets such as the stylesheet and JavaScript file are also exposed explicitly so hosted environments can fetch them reliably.

### Health endpoint

The health endpoint is intended for deployment checks and uptime verification. It reports:

- API status
- Version
- Number of loaded regression/classification models
- Number of loaded clustering models

### Prediction endpoint

The GDP prediction endpoint accepts a structured request payload and returns:

- Predicted value
- Confidence value or supporting metric
- Model used
- Echoed input features

When a requested model is unavailable, the endpoint falls back to the best model recorded in the latest training summary.

### OpenAPI docs

FastAPI automatically exposes interactive docs at `/docs`.

## Frontend Overview

The frontend is a custom static experience located in the `frontend/` directory.

### Files

- [frontend/index.html](frontend/index.html)
- [frontend/app.js](frontend/app.js)
- [frontend/styles.css](frontend/styles.css)

### What the frontend provides

- Hero section and dashboard-style presentation
- Prediction forms and model selectors
- Visual model insights
- Navigation anchored to key sections
- Animated and responsive UI behavior

### Deployment detail

Because the frontend is served from the backend app, the root URL now renders the actual page rather than API JSON. This is important for Hugging Face Spaces and similar deployments.

## Local Setup

This section assumes you are working on Windows, but the same general process applies on macOS and Linux.

### 1. Install prerequisites

Make sure you have:

- Python 3.11 or compatible version
- Docker Desktop if you plan to use containerized runs
- Git
- Optional: Prefect CLI for workflow orchestration

### 2. Clone the repository

If you do not already have the project locally:

```bash
git clone <repo-url>
cd Machine-Learning-AI-221-project
```

### 3. Install Python dependencies

The project uses `requirements.txt` for the main runtime and training dependencies.

```bash
pip install -r requirements.txt
```

### 4. Verify the dataset

Confirm that `countries of the world.csv` exists in the repository root.

### 5. Run the app locally

You can start the API and frontend using Docker or directly through Python.

## Running with Docker

Docker is the recommended local runtime because it mirrors production closely.

### Build and run

```bash
docker compose up --build
```

### Expected endpoints

- Health: http://localhost:8000/health
- API docs: http://localhost:8000/docs
- Frontend: http://localhost:8000/

### Why Docker is useful here

The Docker image is configured to:

- Install project dependencies
- Copy source files and the dataset
- Copy the frontend assets
- Load models and pipeline summaries when present
- Bind to the platform-provided `PORT` value in hosted environments

### Container notes

The container startup command uses dynamic port binding so it works correctly on platforms that inject a runtime port, including Heroku-style container deployments.

## Prefect Workflow

The Prefect workflow is defined in [prefect_workflow.py](prefect_workflow.py).

### Purpose

The workflow automates the full training process:

- Load data
- Preprocess the dataset
- Engineer features
- Create target variables
- Split data into training and test sets
- Train classification models
- Train regression models
- Validate results
- Write pipeline summaries

### How to run

```bash
python prefect_workflow.py
```

### What the workflow produces

- A timestamped summary JSON file in `results/`
- Logs describing training progress
- Performance metrics for the best classification, regression, and clustering models
- Validation output for data quality and quality gates

### Why this matters

The workflow provides a reproducible training path, which is important if you want to regenerate models or compare future improvements against the current baseline.

## Testing and Validation

Testing is a first-class part of this project.

### Test suite

The `tests/` folder includes coverage for:

- API endpoints
- Data loading
- Data validation
- ML model behavior
- Integration checks

### Run the tests

```bash
pytest tests/
```

### Common targeted runs

```bash
pytest tests/test_api.py -v
pytest tests/test_models.py -v
pytest tests/test_data_loader.py -v
pytest tests/test_ml_validation.py -v
```

### Workflow-integrated validation

The project also supports a pipeline that combines model training with validation checks such as:

- Performance thresholds
- Data quality checks
- DeepChecks validation
- Test execution gates

### Why these checks are valuable

They prevent silently degraded deployments. If model quality drops below thresholds, the workflow can stop before bad artifacts are shipped.

## Deployment Guide

The project supports container-based deployment workflows.

### Hugging Face Spaces

The repository is already set up for container-style deployment. The root route serves the frontend, which means the deployed Space shows the interface instead of JSON.

Important deployment settings:

- Use Docker as the deployment backend
- Ensure `HF_TOKEN` has write access
- Ensure the Space repository variable points to the correct Hugging Face repo

### Heroku-style container deployment

The app is also prepared for container deployment on Heroku-like platforms.

Key points:

- The app binds to `${PORT:-8000}`
- The healthcheck uses the same runtime port
- The image includes the frontend and API code
- Placeholder model directories exist so the container build does not fail

### Deployment checklist

Before deployment, verify:

- `Dockerfile` is current
- `frontend/` assets are present
- `results/` contains a recent pipeline summary if you want metrics to show
- `models/` has the expected subdirectories
- The root route serves the frontend

### Root behavior after deployment

The deployed app should show the HTML page at `/`. The API metadata is available at `/api`.

## Artifacts and Outputs

### Results directory

The `results/` folder stores JSON summaries such as:

- `pipeline_summary_*.json`
- `hyperparameters.json`

These files are useful for inspecting training history and comparing runs.

### Model directories

The trained `.pkl` files are expected under:

- `models/regression/`
- `models/classification/`
- `models/clustering/`

### Notebook and legacy files

The `Old-Version/` directory contains previous notebook-oriented assets and should generally be treated as historical reference material.

## Troubleshooting

### 1. The root URL shows JSON instead of the frontend

Make sure [src/api/main.py](src/api/main.py) is serving [frontend/index.html](frontend/index.html) at `/` and that the frontend assets are exposed correctly.

### 2. No models are loaded at startup

This is usually because the `.pkl` files are not present in the container image or deployment environment. Check the model folders and ensure the build process includes them.

### 3. API logs show Pydantic warnings about `model_used`

The response schemas in [src/api/schemas.py](src/api/schemas.py) are configured to avoid protected namespace conflicts. If you edit these models, keep that configuration in place.

### 4. Prefect workflow cannot find the CSV

Verify that `countries of the world.csv` exists in the repository root and that the workflow is pointed at the right path.

### 5. Container build fails on missing model directories

Ensure the placeholder files in `models/` are present so Docker can copy the directory tree even before trained artifacts are generated.

### 6. Tests are slow or noisy

Run only the file you need, for example:

```bash
pytest tests/test_api.py -v
```

### 7. Heroku CLI on Windows does not run from `heroku`

If needed, use the full executable path:

```powershell
& "C:\Program Files\Heroku\bin\heroku.cmd" --version
```

## Maintenance Notes

### Keep these files in sync

If you change the API contract, update:

- [src/api/main.py](src/api/main.py)
- [src/api/schemas.py](src/api/schemas.py)
- [tests/test_api.py](tests/test_api.py)

If you change the frontend entry point, update:

- [frontend/index.html](frontend/index.html)
- [frontend/app.js](frontend/app.js)
- [frontend/styles.css](frontend/styles.css)

If you change model training logic, update:

- [src/models/training.py](src/models/training.py)
- [src/models/classification_models.py](src/models/classification_models.py)
- [src/models/regression_models.py](src/models/regression_models.py)
- [src/models/clustering_models.py](src/models/clustering_models.py)

### Recommended release habit

When you update the training pipeline, regenerate the `results/` summaries so the API can reflect the latest performance numbers.

### Good repository hygiene

- Keep generated `.pkl` files out of version control if they are large or machine-specific
- Retain directory placeholders so container builds remain stable
- Update tests whenever behavior changes
- Prefer small, verifiable changes to deployment code

## Quick Start Summary

If you just want the shortest path to run the project locally:

```bash
pip install -r requirements.txt
docker compose up --build
```

Then open:

- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/docs

## Credits

This project was built as part of an ML/AI coursework workflow focused on country-level economic analysis, model training, workflow automation, and deployable ML applications.

The project structure reflects an emphasis on reproducibility, validation, and practical deployment rather than a notebook-only prototype.
