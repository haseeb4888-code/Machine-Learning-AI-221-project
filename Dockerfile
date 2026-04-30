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
	&& python scripts/bake_models.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
	CMD python -c "import requests; r=requests.get('http://localhost:8000/health', timeout=5); r.raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
