FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV CORS_ORIGINS=https://mahmoudalsalahy.github.io

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models ./models
COPY data ./data

ENV MODEL_PATH=/app/models/model-v1.pkl
ENV XAI_DATA_PATH=/app/data/xai-data.pkl

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
