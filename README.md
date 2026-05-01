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

GitHub Actions runs code checks + tests (including DeepChecks) on pushes/PRs.

### Hugging Face deployment (Spaces)

The pipeline can auto-deploy to a Hugging Face Space on pushes to `main`.

Configure these in your GitHub repo settings:
- **Secret**: `HF_TOKEN` (a Hugging Face access token with write access to the Space)
- **Variable**: `HF_SPACE_REPO` (format: `<namespace>/<space_name>`, e.g. `my-username/my-space`)

## Models

The API loads pickled models from:
- `models/regression/*.pkl`
- `models/classification/*.pkl`

Note: `*.pkl` files are ignored by git (build artifacts). For deployments, the Docker image build step bakes the required Random Forest models into the image.
