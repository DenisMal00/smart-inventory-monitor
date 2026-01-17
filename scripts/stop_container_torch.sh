#!/usr/bin/env bash

CONTAINER_NAME="inventory-torch"

echo "Stopping and removing container: $CONTAINER_NAME"
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || {
  echo "Container not found."
}

echo "Done."
