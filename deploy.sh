#!/usr/bin/env bash
set -euo pipefail

# --- config ---
IMAGE="${DOCKERHUB_USERNAME:-<your-dockerhub-username>}/iris-api:latest"
CONTAINER_NAME="iris-api"
HOST_PORT="${PORT:-8000}"          # you hit http://localhost:8000
MODEL_PATH_IN_CNTR="/app/model/best_iris_model.pkl"
# ---------------

echo "Pulling image: $IMAGE"
docker pull "$IMAGE"

echo "Stopping/removing old container (if any)…"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting new container…"
docker run -d --name "$CONTAINER_NAME" \
  -e MODEL_PATH="$MODEL_PATH_IN_CNTR" \
  -p "$HOST_PORT:80" \
  "$IMAGE"

echo "Showing last 20 lines of logs:"
docker logs --tail=20 "$CONTAINER_NAME"