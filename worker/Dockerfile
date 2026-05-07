FROM python:3.11-slim

WORKDIR /app

# psycopg2 빌드 + 디버깅용 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/ ./worker/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "worker.analysis_worker"]
