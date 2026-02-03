FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY ../requirements/requirements-onnx-container.txt .

RUN pip install --no-cache-dir -r requirements-onnx-container.txt

COPY ../src/app .

COPY ../models/production ./models/production/


ENV PYTHONUNBUFFERED=1
ENV MODEL_MODE=standard

EXPOSE 8000

CMD ["uvicorn","main:app", "--host", "0.0.0.0", "--port", "8000"]