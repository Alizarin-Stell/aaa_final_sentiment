FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV UVICORN_CMD "uvicorn main:app --host 0.0.0.0 --port 8000"

CMD ["sh", "-c", "$UVICORN_CMD"]