#!/bin/bash

# Script to run benchy pipeline in detached mode with proper logging

CONFIG_FILE="${1:-configs/test-gemma.yaml}"
LOG_FILE="logs/detached_$(date +%Y%m%d_%H%M%S).log"

# Ensure logs directory exists
mkdir -p logs

echo "Starting benchy pipeline detached..."
echo "Config: $CONFIG_FILE"
echo "Log file: $LOG_FILE"
echo "Dashboard: http://127.0.0.1:8237"

# Kill any existing vLLM processes
echo "Cleaning up any existing vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Activate virtual environment and run
source .venv/bin/activate

# Run in background with proper logging
nohup python main.py -c "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Pipeline started with PID: $PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Stop with: kill $PID"

# Save PID for cleanup
echo $PID > logs/pipeline.pid

# Function to cleanup on script exit
cleanup() {
    echo "Cleaning up..."
    if [ -f logs/pipeline.pid ]; then
        SAVED_PID=$(cat logs/pipeline.pid)
        kill $SAVED_PID 2>/dev/null || true
        rm -f logs/pipeline.pid
    fi
    # Kill any remaining vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
}

# Set up cleanup on script exit
trap cleanup EXIT

# Show initial logs and monitor for a bit
echo "Initial log output:"
echo "===================="
sleep 3

# Show logs and monitor for signs of actual execution
for i in {1..10}; do
    if [ -f "$LOG_FILE" ]; then
        echo "--- Logs at $(date) ---"
        tail -n 10 "$LOG_FILE"
        
        # Check if we see step execution
        if grep -q "Step.*has started" "$LOG_FILE"; then
            echo "✅ Pipeline steps are executing..."
            break
        elif grep -q "completed successfully" "$LOG_FILE"; then
            echo "⚠️  Pipeline completed very quickly - possibly using cache"
            echo "Last 20 lines:"
            tail -n 20 "$LOG_FILE"
            break
        fi
    fi
    sleep 2
done

echo ""
echo "Use 'tail -f $LOG_FILE' to monitor progress"
echo "Or check the ZenML dashboard: http://127.0.0.1:8237"
