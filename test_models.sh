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

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "🧪 Test Models Script"
    echo "===================="
    echo ""
    echo "Usage: ./test_models.sh [config_name.yaml]"
    echo ""
    echo "This script tests all model configurations in configs/single_card/ by:"
    echo "  • Finding all .yaml files in configs/single_card/"
    echo "  • Running vLLM server tests for each model"
    echo "  • Providing detailed pass/fail summary with model names"
    echo ""
    echo "Arguments:"
    echo "  config_name.yaml    Test only this specific config file (optional)"
    echo ""
    echo "Features:"
    echo "  • Automatic discovery of all single card configs"
    echo "  • Fast testing with --test and --no-log-samples flags"
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
    exit 0
fi

echo "🧪 Starting batch testing of all single card models"
echo "=================================================="

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
    
    # Run the test with --test flag (and --no-log-samples to speed up testing)
    if python eval.py -c "$config" --test --no-log-samples; then
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
