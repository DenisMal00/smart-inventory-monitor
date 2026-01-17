#!/usr/bin/env bash
set -e

CONTAINER_NAME="inventory-container"
IMAGE_NAME="inventory-api:v1"

echo "Removing existing container (if any)..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo "Running container..."
docker run -d \
  -p 8000:8000 \
  --memory="2g" \
  --cpus="1.0" \
  -e MODEL_MODE="standard" \
  --name $CONTAINER_NAME \
  $IMAGE_NAME

echo "Inventory API is running on http://localhost:8000"
