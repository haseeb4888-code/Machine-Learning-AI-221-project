FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
		PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_DEFAULT_TIMEOUT=120 \
	PIP_PROGRESS_BAR=off

COPY requirements-docker.txt ./requirements-docker.txt
RUN python -m pip install --upgrade pip \
	&& pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-docker.txt

COPY . .

# Pre-train and bake required models into the image so deployments (e.g. Hugging Face)
# have models available even without a host-mounted ./models directory.
RUN python -m pip install --no-cache-dir --retries 10 --timeout 120 pandas==2.0.0 \
	&& python -c "from pathlib import Path; from src.models.training import TrainingPipeline; from src.models.classification_models import ClassificationModelManager; from src.models.regression_models import RegressionModelManager; candidates=[Path('Old-Version')/'countries of the world.csv', Path('old-Version')/'countries of the world.csv']; data_path=next((p for p in candidates if p.exists()), None);\
assert data_path is not None, f'CSV not found. Tried: {[str(p) for p in candidates]}';\
p=TrainingPipeline(str(data_path)); p.load_and_preprocess(); p.engineer_features(); p.create_target_variable(); p.prepare_train_test_split();\
c=ClassificationModelManager(); c.train_random_forest(p.X_train, p.y_train_clf, p.X_test, p.y_test_clf); c.save_models();\
r=RegressionModelManager(); r.train_random_forest_regressor(p.X_train, p.y_train_reg, p.X_test, p.y_test_reg); r.save_models();\
print('✓ Baked models into image')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
	CMD python -c "import requests; r=requests.get('http://localhost:8000/health', timeout=5); r.raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
