#!/bin/bash
# Batch evaluation script to run multiple models sequentially
# Discovers config files automatically or accepts a list file
#
# Usage: ./run_models.sh [--help] [--quiet] [config_list.txt] [config_name.yaml]
#
# This script will:
# 1. Find all .yaml files in configs/models/ (or use provided list/file)
# 2. Run 'benchy eval -c <config>' for each model
# 3. Provide detailed summary with lists of passed/failed models

# Handle command line flags
QUIET_MODE="false"
CUSTOM_RUN_ID=""
SUBFOLDER=""
TASKS_OVERRIDE=""
TASKS_FILE=""
TASK_GROUPS=()
POSITIONAL_ARGS=()

# Prefer installed `benchy`, fall back to running from the repo venv/module.
if command -v benchy >/dev/null 2>&1; then
    BENCHY_CMD=(benchy)
elif [[ -x ".venv/bin/python" ]]; then
    BENCHY_CMD=(.venv/bin/python -m src.benchy_cli)
else
    BENCHY_CMD=(python -m src.benchy_cli)
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "🚀 Run Models Script"
            echo "==================="
            echo ""
            echo "Usage: ./run_models.sh [--quiet] [--run-id ID] [--subfolder FOLDER] [--tasks TASKS] [--tasks-file FILE] [--task-group GROUP] [config_list.txt] [config_name.yaml]"
            echo ""
            echo "This script runs evaluation on multiple model configurations by:"
            echo "  • Finding all .yaml files in configs/models/ (default)"
            echo "  • Optionally targeting a specific subfolder (e.g., configs/models/pending/)"
            echo "  • Running full evaluation for each model"
            echo "  • Providing detailed pass/fail summary with model names"
            echo ""
            echo "Arguments:"
            echo "  config_list.txt      Text file containing list of config files (one per line)"
            echo "  config_name.yaml     Run only this specific config file (optional)"
            echo ""
            echo "Config List File Format:"
            echo "  The config_list.txt file should contain one config file per line."
            echo "  You can use either relative or absolute paths:"
            echo ""
            echo "  Relative paths (recommended):"
            echo "    together_model-A.yaml"
            echo "    together_model-B.yaml"
            echo "    # Comments start with #"
            echo "    # Empty lines are ignored"
            echo ""
            echo "  Absolute paths (also supported):"
            echo "    /path/to/configs/models/model-A.yaml"
            echo "    /path/to/configs/models/model-B.yaml"
            echo ""
            echo "  When using relative paths, they are resolved relative to:"
            echo "    • configs/models/ (default)"
            echo "    • configs/models/SUBFOLDER/ (if --subfolder is specified)"
            echo ""
            echo "  Tips for generating config list files:"
            echo "    # List all YAML files in a directory:"
            echo "    ls configs/models/*.yaml | xargs -n1 basename > my_list.txt"
            echo ""
            echo "    # List specific models (using grep):"
            echo "    ls configs/models/*.yaml | xargs -n1 basename | grep 'together_' > my_list.txt"
            echo ""
            echo "    # List models from a subfolder:"
            echo "    ls configs/models/pending/*.yaml | xargs -n1 basename > pending_list.txt"
            echo ""
            echo "    # Manual editing - just add one filename per line"
            echo ""
            echo "Options:"
            echo "  --quiet             Suppress detailed pipeline output (recommended for long runs)"
            echo "  --run-id ID         Use custom run ID for organizing outputs (default: auto-generated)"
            echo "  --subfolder FOLDER  Run models from a specific subfolder (e.g., pending, completed)"
            echo "  --tasks TASKS       Comma-separated task list (overrides config tasks)"
            echo "  --tasks-file FILE   Task list file (one task per line, overrides config tasks)"
            echo "  --task-group GROUP  Task group name from configs/config.yaml (can repeat)"
            echo ""
            echo "Features:"
            echo "  • Automatic discovery of all model configs"
            echo "  • Subfolder targeting for organized model groups"
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
            echo "  # Run all models in configs/models/"
            echo "  ./run_models.sh"
            echo ""
            echo "  # Run all models in configs/models/pending/"
            echo "  ./run_models.sh --subfolder pending"
            echo ""
            echo "  # Run all models quietly"
            echo "  ./run_models.sh --quiet"
            echo ""
            echo "  # Run with custom run ID"
            echo "  ./run_models.sh --run-id my_experiment_001"
            echo ""
            echo "  # Run pending models with custom run ID"
            echo "  ./run_models.sh --subfolder pending --run-id pending_batch_001"
            echo ""
            echo "  # Run specific model"
            echo "  ./run_models.sh my-model.yaml"
            echo ""
            echo "  # Run models from list file (relative paths)"
            echo "  ./run_models.sh my_model_list.txt"
            echo ""
            echo "  # Run models from list file with absolute paths"
            echo "  ./run_models.sh /path/to/my_model_list.txt"
            echo ""
            echo "  # Run all models with a shared task list"
            echo "  ./run_models.sh --tasks spanish,portuguese,translation"
            echo ""
            echo "  # Run all models with a task list file"
            echo "  ./run_models.sh --tasks-file configs/tests/task_list.txt"
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
        --subfolder)
            SUBFOLDER="$2"
            shift 2
            ;;
        --tasks)
            TASKS_OVERRIDE="$2"
            shift 2
            ;;
        --tasks-file)
            TASKS_FILE="$2"
            shift 2
            ;;
        --task-group)
            TASK_GROUPS+=("$2")
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
echo "   Outputs will be organized under: outputs/benchmark_outputs/$RUN_ID"
echo "   Logs will be organized under: logs/$RUN_ID"
echo ""

if [[ "$QUIET_MODE" == "true" ]]; then
    echo "🔇 Quiet mode enabled - suppressing detailed pipeline output"
    echo "   Individual model results will only show pass/fail status"
    echo "   Full logs are still saved by the pipeline to individual files"
    echo ""
fi

if [[ -n "$TASKS_OVERRIDE" || -n "$TASKS_FILE" || ${#TASK_GROUPS[@]} -gt 0 ]]; then
    echo "🧭 Task overrides enabled for this batch:"
    if [[ -n "$TASKS_OVERRIDE" ]]; then
        echo "   --tasks $TASKS_OVERRIDE"
    fi
    if [[ -n "$TASKS_FILE" ]]; then
        echo "   --tasks-file $TASKS_FILE"
    fi
    for group in "${TASK_GROUPS[@]}"; do
        echo "   --task-group $group"
    done
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
if [[ -n "$SUBFOLDER" ]]; then
    config_dir="./configs/models/$SUBFOLDER"
    echo "📁 Using subfolder: $SUBFOLDER"
    echo "   Target directory: $config_dir"
else
    config_dir="./configs/models"
fi

if [[ -n "$1" ]]; then
    # Check if it's a text file with config list
    if [[ -f "$1" && "$1" == *.txt ]]; then
        echo "📋 Loading config list from: $1"
        # Read config files from the text file
        # Handle both absolute paths and relative paths (prepend config_dir only if relative)
        mapfile -t configs < <(grep -v '^#' "$1" | grep -v '^$' | while IFS= read -r line; do
            if [[ "$line" == /* ]]; then
                # Absolute path - use as-is
                echo "$line"
            else
                # Relative path - prepend config_dir
                echo "$config_dir/$line"
            fi
        done)
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
    # Default: discover all configs in models directory (or subfolder)
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
 
# Build eval args for task overrides
eval_args=()
if [[ -n "$TASKS_OVERRIDE" ]]; then
    eval_args+=(--tasks "$TASKS_OVERRIDE")
fi
if [[ -n "$TASKS_FILE" ]]; then
    eval_args+=(--tasks-file "$TASKS_FILE")
fi
if [[ ${#TASK_GROUPS[@]} -gt 0 ]]; then
    for group in "${TASK_GROUPS[@]}"; do
        eval_args+=(--task-group "$group")
    done
fi

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
        "${BENCHY_CMD[@]}" eval -c "$config" --run-id "$RUN_ID" "${eval_args[@]}" > /dev/null 2>&1
        eval_result=$?
    else
        # Normal mode: show full output
        "${BENCHY_CMD[@]}" eval -c "$config" --run-id "$RUN_ID" "${eval_args[@]}"
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
