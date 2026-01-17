#!/usr/bin/env bash

IMAGE_NAME="inventory-api:torch"
CONTAINER_NAME="inventory-torch"
DOCKERFILE="Dockerfile.torch"

echo "Removing existing container (if any)..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

echo "Building image: $IMAGE_NAME"
docker build -t $IMAGE_NAME -f $DOCKERFILE .

echo "Starting container: $CONTAINER_NAME"
docker run -d \
  -p 8000:8000 \
  --memory="2g" \
  --cpus="1.0" \
  --name $CONTAINER_NAME \
  $IMAGE_NAME

echo "Inventory API is running on http://localhost:8000"
