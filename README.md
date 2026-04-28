---
title: Machine Learning AI-221 Project
sdk: docker
app_port: 8000
---

# Machine-Learning-AI-221-project

## Local run (Docker)

- Start: `docker compose up --build`
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs

## CI

GitHub Actions runs tests (including DeepChecks) on pushes/PRs to `main`.

## Models

The API loads pickled models from:
- `models/regression/*.pkl`
- `models/classification/*.pkl`

Note: `*.pkl` files are ignored by git (build artifacts). For deployments, the Docker image build step bakes the required Random Forest models into the image.
