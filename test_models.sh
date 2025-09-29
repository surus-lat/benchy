#!/bin/bash
# Test script to validate all single card model configurations
# Tests all .yaml files in configs/single_card with --test flag
#
# Usage: ./test_models.sh [--help] [config_name.yaml]
#
# This script will:
# 1. Find all .yaml files in configs/single_card/ (or test just one if specified)
# 2. Run 'python eval.py -c <config> --test --no-log-samples' for each
# 3. Provide detailed summary with lists of passed/failed models

# Handle command line flags
QUIET_MODE="false"
LIMITED_MODE="false"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "🧪 Test Models Script"
            echo "===================="
            echo ""
            echo "Usage: ./test_models.sh [--quiet] [--limited] [config_name.yaml]"
            echo ""
            echo "This script tests all model configurations in configs/single_card/ by:"
            echo "  • Finding all .yaml files in configs/single_card/"
            echo "  • Running vLLM server tests for each model"
            echo "  • Providing detailed pass/fail summary with model names"
            echo ""
            echo "Arguments:"
            echo "  config_name.yaml    Test only this specific config file (optional)"
            echo ""
            echo "Options:"
            echo "  --quiet             Suppress detailed pipeline output (recommended for long runs)"
            echo "  --limited           Run limited evaluation with --log-samples --limit 10 instead of test mode"
            echo ""
            echo "Features:"
            echo "  • Automatic discovery of all single card configs"
            echo "  • Fast testing with --test and --no-log-samples flags (default)"
            echo "  • Limited evaluation with --log-samples --limit 10 (--limited mode)"
            echo "  • Detailed summary showing which models passed/failed"
            echo "  • Exit code 0 if all pass, 1 if any fail"
            echo ""
            echo "Example output:"
            echo "  ✅ MODELS THAT PASSED TESTING (15):"
            echo "    ✓ ByteDance-Seed/Seed-X-Instruct-7B"
            echo "      └── Config: Seed-X-Instruct-7B.yaml"
            echo ""
            echo "  ❌ MODELS THAT FAILED TESTING (2):"
            echo "    ✗ Some/Broken-Model"
            echo "      └── Config: broken-model.yaml"
            echo ""
            echo "Usage with nohup:"
            echo "  # Quick test mode (default)"
            echo "  nohup ./test_models.sh --quiet > test_results.log 2>&1 &"
            echo ""
            echo "  # Limited evaluation mode"
            echo "  nohup ./test_models.sh --limited --quiet > limited_results.log 2>&1 &"
            echo ""
            echo "  # Limited evaluation on specific model"
            echo "  nohup ./test_models.sh --limited --quiet my-model.yaml > my_model_limited.log 2>&1 &"
            echo ""
            exit 0
            ;;
        --quiet|-q)
            QUIET_MODE="true"
            shift
            ;;
        --limited|-l)
            LIMITED_MODE="true"
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# This section is now handled by the while loop above

echo "🧪 Starting batch testing of all single card models"
echo "=================================================="

if [[ "$QUIET_MODE" == "true" ]]; then
    echo "🔇 Quiet mode enabled - suppressing detailed pipeline output"
    echo "   Individual test results will only show pass/fail status"
    echo "   Full logs are still saved by the pipeline to individual files"
    echo ""
fi

if [[ "$LIMITED_MODE" == "true" ]]; then
    echo "📊 Limited mode enabled - running evaluation with --log-samples --limit 10"
    echo "   This will run actual evaluation on 10 samples instead of quick test mode"
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

# Find all .yaml config files in configs/single_card
config_dir="./configs/single_card"
if [[ ! -d "$config_dir" ]]; then
    echo "❌ Directory not found: $config_dir"
    exit 1
fi

# Check if user specified a single config to test
if [[ -n "$1" ]]; then
    # Test only the specified config
    single_config="$config_dir/$1"
    if [[ ! -f "$single_config" ]]; then
        echo "❌ Config file not found: $single_config"
        echo "Available configs in $config_dir:"
        ls -1 "$config_dir"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/^/  - /'
        exit 1
    fi
    configs=("$single_config")
    echo "🎯 Testing single config: $1"
else
    # Get all .yaml files
    mapfile -t configs < <(find "$config_dir" -name "*.yaml" | sort)
fi

if [[ ${#configs[@]} -eq 0 ]]; then
    echo "❌ No .yaml files found in $config_dir"
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

echo "Found $total model configurations to test:"
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

# Run tests for each model
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
    
    echo "🔄 Testing model $model_num/$total: $model_name"
    echo "   Config: $config_name"
    echo "   Started at: $(date)"
    
    start_time=$(date +%s)
    
    # Run the test with appropriate flags based on mode
    # Redirect output to suppress verbose logging (pipeline logs to its own files)
    if [[ "$LIMITED_MODE" == "true" ]]; then
        # Limited mode: run actual evaluation with --log-samples --limit 10
        if [[ "$QUIET_MODE" == "true" ]]; then
            python eval.py -c "$config" --log-samples --limit 10 > /dev/null 2>&1
            test_result=$?
        else
            python eval.py -c "$config" --log-samples --limit 10
            test_result=$?
        fi
    else
        # Default mode: run quick test with --test flag (and --no-log-samples to speed up testing)
        if [[ "$QUIET_MODE" == "true" ]]; then
            python eval.py -c "$config" --test --no-log-samples > /dev/null 2>&1
            test_result=$?
        else
            python eval.py -c "$config" --test --no-log-samples
            test_result=$?
        fi
    fi
    
    if [[ $test_result -eq 0 ]]; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✅ Model $model_num test PASSED in ${duration}s"
        passed_models+=("$model_name")
        passed_configs+=("$config_name")
        ((successful++))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "❌ Model $model_num test FAILED after ${duration}s"
        failed_models+=("$model_name")
        failed_configs+=("$config_name")
        ((failed++))
    fi
    
    echo "   Finished at: $(date)"
    echo ""
done

# Killer feature: Detailed summary with model lists
echo "=================================================="
echo "🧪 DETAILED TEST RESULTS SUMMARY"
echo "=================================================="
echo "Total models tested: $total"
echo "Tests passed: $successful"
echo "Tests failed: $failed"
echo ""

if [[ $successful -gt 0 ]]; then
    echo "✅ MODELS THAT PASSED TESTING ($successful):"
    echo "----------------------------------------"
    for i in "${!passed_models[@]}"; do
        model="${passed_models[$i]}"
        config="${passed_configs[$i]}"
        echo "  ✓ $model"
        echo "    └── Config: $config"
    done
    echo ""
fi

if [[ $failed -gt 0 ]]; then
    echo "❌ MODELS THAT FAILED TESTING ($failed):"
    echo "---------------------------------------"
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
    echo "🎉 ALL MODELS PASSED! Ready for production runs."
    exit_code=0
else
    echo "⚠️  Some models failed testing. Check configurations before production runs."
    exit_code=1
fi

echo "Completed at: $(date)"
echo "=================================================="

exit $exit_code
