#!/usr/bin/env bash

CONTAINER_NAME="inventory-container"

echo "Stopping and removing container: $CONTAINER_NAME"

docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || {
  echo "ℹContainer not found. Nothing to remove."
}

echo "Done."
