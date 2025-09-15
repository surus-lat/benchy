#!/bin/bash
set -e

echo "🚀 Starting ZenML server..."

# Check if docker-compose is available
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Check if we need sudo for docker commands
if ! docker ps >/dev/null 2>&1; then
    echo "🔐 Docker requires sudo permissions. Using sudo for docker commands."
    COMPOSE_CMD="sudo $COMPOSE_CMD"
fi

# Remove existing container if it exists
echo "🧹 Cleaning up any existing zenml container..."
$COMPOSE_CMD rm -f zenml 2>/dev/null || true

# Start ZenML server
$COMPOSE_CMD up -d zenml

# Wait for health check
echo "⏳ Waiting for ZenML server to be ready..."
sleep 15

# Check if service is healthy
if $COMPOSE_CMD ps zenml | grep -q "healthy\|Up"; then
    echo "✅ ZenML server is running at http://localhost:8080"
else
    echo "⚠️  ZenML server started but health check pending..."
    echo "   Check status with: $COMPOSE_CMD logs zenml"
fi
