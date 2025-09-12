#!/bin/bash
# Simple shell script to run multiple models sequentially

echo "🚀 Starting batch evaluation of multiple models"
echo "=============================================="

# Array of config files
configs=(
    "configs/model-1-qwen.yaml"
    "configs/model-2-gemma.yaml" 
    "configs/model-3-llama.yaml"
    # Add more config files here
)

# Track results
total=${#configs[@]}
successful=0
failed=0

echo "Planning to evaluate $total models:"
for config in "${configs[@]}"; do
    model_name=$(grep "name:" "$config" | cut -d'"' -f2)
    echo "  - $model_name ($config)"
done
echo ""

# Run each model
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    model_num=$((i + 1))
    
    if [[ ! -f "$config" ]]; then
        echo "❌ Config file not found: $config"
        ((failed++))
        continue
    fi
    
    model_name=$(grep "name:" "$config" | cut -d'"' -f2)
    echo "🔄 Running model $model_num/$total: $model_name"
    echo "   Config: $config"
    echo "   Started at: $(date)"
    
    start_time=$(date +%s)
    
    # Run the evaluation
    if BENCHY_CONFIG="$config" python main.py; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✅ Model $model_num completed successfully in ${duration}s"
        ((successful++))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "❌ Model $model_num failed after ${duration}s"
        ((failed++))
    fi
    
    echo "   Finished at: $(date)"
    echo ""
done

# Summary
echo "=============================================="
echo "📊 BATCH EVALUATION SUMMARY"
echo "=============================================="
echo "Total models: $total"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Completed at: $(date)"
