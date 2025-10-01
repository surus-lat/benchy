#!/bin/bash
# Batch evaluation script to run multiple models sequentially
# Discovers config files automatically or accepts a list file
#
# Usage: ./run_models.sh [--help] [--quiet] [config_list.txt] [config_name.yaml]
#
# This script will:
# 1. Find all .yaml files in configs/single_card/ (or use provided list/file)
# 2. Run 'python eval.py -c <config>' for each model
# 3. Provide detailed summary with lists of passed/failed models

# Handle command line flags
QUIET_MODE="false"
CUSTOM_RUN_ID=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "🚀 Run Models Script"
            echo "==================="
            echo ""
            echo "Usage: ./run_models.sh [--quiet] [--run-id ID] [config_list.txt] [config_name.yaml]"
            echo ""
            echo "This script runs evaluation on multiple model configurations by:"
            echo "  • Finding all .yaml files in configs/single_card/ (default)"
            echo "  • Running full evaluation for each model"
            echo "  • Providing detailed pass/fail summary with model names"
            echo ""
            echo "Arguments:"
            echo "  config_list.txt      Text file containing list of config files (one per line)"
            echo "  config_name.yaml     Run only this specific config file (optional)"
            echo ""
            echo "Options:"
            echo "  --quiet             Suppress detailed pipeline output (recommended for long runs)"
            echo "  --run-id ID         Use custom run ID for organizing outputs (default: auto-generated)"
            echo ""
            echo "Features:"
            echo "  • Automatic discovery of all single card configs"
            echo "  • Support for config list files"
            echo "  • Single config execution"
            echo "  • Automatic run ID generation for organized outputs"
            echo "  • Detailed summary showing which models passed/failed"
            echo "  • Exit code 0 if all pass, 1 if any fail"
            echo ""
            echo "Example output:"
            echo "  ✅ MODELS THAT COMPLETED SUCCESSFULLY (15):"
            echo "    ✓ ByteDance-Seed/Seed-X-Instruct-7B"
            echo "      └── Config: Seed-X-Instruct-7B.yaml"
            echo ""
            echo "  ❌ MODELS THAT FAILED (2):"
            echo "    ✗ Some/Broken-Model"
            echo "      └── Config: broken-model.yaml"
            echo ""
            echo "Usage examples:"
            echo "  # Run all models in configs/single_card/"
            echo "  ./run_models.sh"
            echo ""
            echo "  # Run all models quietly"
            echo "  ./run_models.sh --quiet"
            echo ""
            echo "  # Run with custom run ID"
            echo "  ./run_models.sh --run-id my_experiment_001"
            echo ""
            echo "  # Run specific model"
            echo "  ./run_models.sh my-model.yaml"
            echo ""
            echo "  # Run models from list file"
            echo "  ./run_models.sh my_model_list.txt"
            echo ""
            echo "  # Run with nohup for long sessions"
            echo "  nohup ./run_models.sh --quiet > run_results.log 2>&1 &"
            echo ""
            exit 0
            ;;
        --quiet|-q)
            QUIET_MODE="true"
            shift
            ;;
        --run-id)
            CUSTOM_RUN_ID="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

echo "🚀 Starting batch evaluation of multiple models"
echo "=============================================="

# Generate or use custom run ID for this batch
if [[ -n "$CUSTOM_RUN_ID" ]]; then
    RUN_ID="$CUSTOM_RUN_ID"
    echo "📋 Using custom run ID: $RUN_ID"
else
    RUN_ID="batch_$(date +%Y%m%d_%H%M%S)"
    echo "📋 Generated run ID: $RUN_ID"
fi
echo "   All models in this batch will use the same run ID for organized outputs"
echo ""

if [[ "$QUIET_MODE" == "true" ]]; then
    echo "🔇 Quiet mode enabled - suppressing detailed pipeline output"
    echo "   Individual model results will only show pass/fail status"
    echo "   Full logs are still saved by the pipeline to individual files"
    echo ""
fi

# Activate benchy virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "✅ Activated benchy virtual environment"
else
    echo "❌ Virtual environment not found at .venv/bin/activate"
    exit 1
fi

echo "Current virtual environment: $VIRTUAL_ENV"
echo ""

# Determine which configs to run
config_dir="./configs/single_card"

if [[ -n "$1" ]]; then
    # Check if it's a text file with config list
    if [[ -f "$1" && "$1" == *.txt ]]; then
        echo "📋 Loading config list from: $1"
        # Read config files from the text file
        mapfile -t configs < <(grep -v '^#' "$1" | grep -v '^$' | sed 's|^|./configs/single_card/|')
        echo "Found ${#configs[@]} configs in list file"
    else
        # Single config file specified
        single_config="$config_dir/$1"
        if [[ ! -f "$single_config" ]]; then
            echo "❌ Config file not found: $single_config"
            echo "Available configs in $config_dir:"
            ls -1 "$config_dir"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/^/  - /'
            exit 1
        fi
        configs=("$single_config")
        echo "🎯 Running single config: $1"
    fi
else
    # Default: discover all configs in single_card directory
    if [[ ! -d "$config_dir" ]]; then
        echo "❌ Directory not found: $config_dir"
        exit 1
    fi
    echo "🔍 Auto-discovering configs in: $config_dir"
    mapfile -t configs < <(find "$config_dir" -name "*.yaml" | sort)
    echo "Found ${#configs[@]} config files"
fi

if [[ ${#configs[@]} -eq 0 ]]; then
    echo "❌ No config files found"
    exit 1
fi

# Arrays to track results
declare -a passed_models=()
declare -a failed_models=()
declare -a passed_configs=()
declare -a failed_configs=()

# Track stats
total=${#configs[@]}
successful=0
failed=0

echo "Planning to evaluate $total models:"
for config in "${configs[@]}"; do
    config_name=$(basename "$config")
    if [[ -f "$config" ]]; then
        model_name=$(grep "name:" "$config" | head -1 | sed 's/.*name: *["'\'']\([^"'\'']*\)["'\''].*/\1/')
        if [[ -z "$model_name" ]]; then
            model_name="Unknown"
        fi
        echo "  - $model_name ($config_name)"
    else
        echo "  - Config not found ($config_name)"
    fi
done
echo ""
 
# Run each model
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    config_name=$(basename "$config")
    model_num=$((i + 1))
    
    if [[ ! -f "$config" ]]; then
        echo "❌ Config file not found: $config"
        failed_models+=("Config not found")
        failed_configs+=("$config_name")
        ((failed++))
        continue
    fi
    
    # Extract model name more robustly
    model_name=$(grep "name:" "$config" | head -1 | sed 's/.*name: *["'\'']\([^"'\'']*\)["'\''].*/\1/')
    if [[ -z "$model_name" ]]; then
        model_name=$(basename "$config" .yaml)
    fi
    
    echo "🔄 Running model $model_num/$total: $model_name"
    echo "   Config: $config_name"
    echo "   Started at: $(date)"
    
    start_time=$(date +%s)
    
    # Run the evaluation with appropriate output handling
    if [[ "$QUIET_MODE" == "true" ]]; then
        # Quiet mode: suppress detailed output
        python eval.py -c "$config" --run-id "$RUN_ID" > /dev/null 2>&1
        eval_result=$?
    else
        # Normal mode: show full output
        python eval.py -c "$config" --run-id "$RUN_ID"
        eval_result=$?
    fi
    
    if [[ $eval_result -eq 0 ]]; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✅ Model $model_num completed successfully in ${duration}s"
        passed_models+=("$model_name")
        passed_configs+=("$config_name")
        ((successful++))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "❌ Model $model_num failed after ${duration}s"
        failed_models+=("$model_name")
        failed_configs+=("$config_name")
        ((failed++))
    fi
    
    echo "   Finished at: $(date)"
    echo ""
done

# Detailed summary with model lists
echo "=============================================="
echo "📊 DETAILED EVALUATION RESULTS SUMMARY"
echo "=============================================="
echo "Total models evaluated: $total"
echo "Evaluations completed: $successful"
echo "Evaluations failed: $failed"
echo ""

if [[ $successful -gt 0 ]]; then
    echo "✅ MODELS THAT COMPLETED SUCCESSFULLY ($successful):"
    echo "-----------------------------------------------"
    for i in "${!passed_models[@]}"; do
        model="${passed_models[$i]}"
        config="${passed_configs[$i]}"
        echo "  ✓ $model"
        echo "    └── Config: $config"
    done
    echo ""
fi

if [[ $failed -gt 0 ]]; then
    echo "❌ MODELS THAT FAILED ($failed):"
    echo "-------------------------------"
    for i in "${!failed_models[@]}"; do
        model="${failed_models[$i]}"
        config="${failed_configs[$i]}"
        echo "  ✗ $model"
        echo "    └── Config: $config"
    done
    echo ""
fi

# Overall result
if [[ $failed -eq 0 ]]; then
    echo "🎉 ALL MODELS COMPLETED SUCCESSFULLY! Evaluation batch finished."
    exit_code=0
else
    echo "⚠️  Some models failed evaluation. Check logs for details."
    exit_code=1
fi

echo "Completed at: $(date)"
echo "=============================================="

exit $exit_code
