FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        fastapi \
        uvicorn \
        requests \
        python-multipart \
        httpx

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8444"]