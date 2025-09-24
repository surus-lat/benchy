#!/usr/bin/env python3
"""
Script to parse raw model results and generate summary data for each model.
This processes the JSON results files and extracts hierarchical scores.

MODULAR STRUCTURE:
- Each task has its own processing function (e.g., process_spanish_results, process_portuguese_results)
- Task-specific functions use task configs to determine output paths
- Common score extraction logic is shared via extract_metric_from_task_data
- New tasks can be easily added by:
  1. Creating a process_<task>_results function
  2. Adding it to get_available_task_processors()
  3. Creating a task config file in configs/tasks/
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_task_config(task_name: str) -> Dict:
    """Load task-specific configuration from YAML file."""
    # Try multiple possible paths for the config
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / "tasks" / f"{task_name}.yaml",
        Path("configs") / "tasks" / f"{task_name}.yaml",
        Path("benchy") / "configs" / "tasks" / f"{task_name}.yaml",
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    
    return {}

def extract_metric_from_task_data(task_data: Dict) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Extract main metric, score, and stderr from task data."""
    main_metric = None
    main_score = None
    main_stderr = None
    
    # Look for "acc,all" first (Portuguese format), then "acc,none" (Spanish format)
    for key, value in task_data.items():
        if key == "acc,all":
            main_metric = "acc"
            main_score = value
            main_stderr = task_data.get("acc_stderr,all", 0.0)
            break
        elif key == "acc,none":
            main_metric = "acc"
            main_score = value
            main_stderr = task_data.get("acc_stderr,none", 0.0)
            break
        elif key.endswith(",all") and not key.endswith("_stderr,all"):
            metric_name = key.replace(",all", "")
            if metric_name in ["acc", "exact_match", "acc_norm", "f1_macro"]:
                main_metric = metric_name
                main_score = value
                stderr_key = f"{metric_name}_stderr,all"
                main_stderr = task_data.get(stderr_key, 0.0)
                break
        elif key.endswith(",none") and not key.endswith("_stderr,none"):
            metric_name = key.replace(",none", "")
            if metric_name in ["acc", "exact_match", "acc_norm", "f1_macro"]:
                main_metric = metric_name
                main_score = value
                stderr_key = f"{metric_name}_stderr,none"
                main_stderr = task_data.get(stderr_key, 0.0)
                break
    
    return main_metric, main_score, main_stderr

def extract_scores_from_results(results_data: Dict) -> Dict[str, Any]:
    """Extract and organize scores from the new benchmark results format."""
    
    # Extract individual task scores
    task_scores = {}
    for task_name, task_data in results_data.get("results", {}).items():
        if not task_name.startswith("latam_"):  # Skip top-level categories
            main_metric, main_score, main_stderr = extract_metric_from_task_data(task_data)
            
            if main_score is not None:
                task_scores[task_name] = {
                    "score": main_score,
                    "stderr": main_stderr,
                    "metric": main_metric,
                    "alias": task_name
                }
    
    # Extract category scores (latam_pr, latam_es, spanish, teleia, etc.)
    category_scores = {}
    for category_name, category_data in results_data.get("results", {}).items():
        if category_name.startswith("latam_") or category_name in ["spanish", "teleia"]:
            main_metric, main_score, main_stderr = extract_metric_from_task_data(category_data)
            
            if main_score is not None:
                category_scores[category_name] = {
                    "score": main_score,
                    "stderr": main_stderr,
                    "metric": main_metric,
                    "alias": category_name
                }
    
    # Calculate overall LATAM score (average of all category scores)
    overall_score = None
    if category_scores:
        scores = [data["score"] for data in category_scores.values()]
        overall_score = sum(scores) / len(scores)
    
    return {
        "task_scores": task_scores,
        "category_scores": category_scores,
        "top_level_scores": category_scores,  # Same as category_scores in new format
        "overall_score": overall_score,
        "model_name": "unknown",  # Will be determined from directory name
        "model_name_sanitized": "unknown",
        "evaluation_time": "unknown"
    }

def load_tasks_mapping() -> Dict[str, Any]:
    """Load the tasks mapping from tasks.json."""
    tasks_file = Path(__file__).parent.parent / "tasks.json"
    if tasks_file.exists():
        with open(tasks_file, 'r') as f:
            return json.load(f)
    return {}

def process_spanish_results(model_dir: Path, model_name: str) -> Optional[Dict[str, Any]]:
    """Process Spanish evaluation results for a model."""
    print(f"    Processing Spanish results...")
    
    # Load Spanish task config to get output subdirectory
    spanish_config = load_task_config("spanish")
    output_subdir = spanish_config.get("output", {}).get("subdirectory", "spanish")
    
    # Look for Spanish results in the subdirectory
    spanish_dir = model_dir / output_subdir
    if not spanish_dir.exists():
        print(f"    Warning: Spanish results directory not found: {spanish_dir}")
        return None
    
    # Look for results_*.json files in Spanish directory (including nested subdirectories)
    results_files = list(spanish_dir.rglob("results_*.json"))
    if not results_files:
        print(f"    Warning: No results_*.json files found in {spanish_dir}")
        return None
    
    # Process the first results file found
    results_file = results_files[0]
    try:
        with open(results_file, 'r') as f:
            spanish_results_data = json.load(f)
        
        spanish_scores = extract_scores_from_results(spanish_results_data)
        spanish_scores["model_name"] = model_name
        
        print(f"    ✓ Spanish results processed from {results_file.relative_to(model_dir)}")
        return spanish_scores
        
    except Exception as e:
        print(f"    Error processing Spanish results from {results_file.name}: {e}")
        return None

def process_portuguese_results(model_dir: Path, model_name: str) -> Optional[Dict[str, Any]]:
    """Process Portuguese evaluation results for a model."""
    print(f"    Processing Portuguese results...")
    
    # Load Portuguese task config to get output subdirectory
    portuguese_config = load_task_config("portuguese")
    output_subdir = portuguese_config.get("output", {}).get("subdirectory", "portuguese")
    
    # Look for Portuguese results in the subdirectory
    portuguese_dir = model_dir / output_subdir
    if not portuguese_dir.exists():
        print(f"    Warning: Portuguese results directory not found: {portuguese_dir}")
        return None
    
    # Look for results.json in Portuguese directory
    results_file = portuguese_dir / "results.json"
    if not results_file.exists():
        print(f"    Warning: No results.json found in {portuguese_dir}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            portuguese_results_data = json.load(f)
        
        portuguese_scores = extract_scores_from_results(portuguese_results_data)
        portuguese_scores["model_name"] = model_name
        
        print(f"    ✓ Portuguese results processed from {results_file.name}")
        return portuguese_scores
        
    except Exception as e:
        print(f"    Error processing Portuguese results from {results_file.name}: {e}")
        return None

def process_translation_results(model_dir: Path, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Template function for processing translation evaluation results.
    Copy this function and modify for new tasks.
    """
    print(f"    Processing Translation results...")
    
    # Load task config to get output subdirectory
    task_config = load_task_config("translation")
    output_subdir = task_config.get("output", {}).get("subdirectory", "translation")
    
    # Look for results in the subdirectory
    task_dir = model_dir / output_subdir
    if not task_dir.exists():
        print(f"    Warning: Translation results directory not found: {task_dir}")
        return None
    
    # Look for results files (modify pattern as needed)
    results_files = list(task_dir.glob("results*.json"))
    if not results_files:
        print(f"    Warning: No results files found in {task_dir}")
        return None
    
    # Process the first results file found
    results_file = results_files[0]
    try:
        with open(results_file, 'r') as f:
            task_results_data = json.load(f)
        
        task_scores = extract_scores_from_results(task_results_data)
        task_scores["model_name"] = model_name
        
        print(f"    ✓ Translation results processed from {results_file.name}")
        return task_scores
        
    except Exception as e:
        print(f"    Error processing Translation results from {results_file.name}: {e}")
        return None

def get_available_task_processors() -> Dict[str, callable]:
    """Get dictionary of available task processors."""
    return {
        "spanish": process_spanish_results,
        "portuguese": process_portuguese_results,
        # Add new tasks here as they are implemented
        # "translation": process_translation_results,
    }

def process_model_directory(model_dir: Path) -> Dict[str, Any]:
    """Process model results using task-specific functions."""
    model_name = model_dir.name
    print(f"  Processing {model_name}...")
    
    try:
        # Load tasks mapping
        tasks_mapping = load_tasks_mapping()
        
        # Get available task processors
        task_processors = get_available_task_processors()
        
        # Process each task using task-specific functions
        all_scores = {}
        
        for task_name, processor_func in task_processors.items():
            task_scores = processor_func(model_dir, model_name)
            if task_scores:
                all_scores[task_name] = task_scores
        
        if not all_scores:
            print(f"  Warning: No valid results found for {model_name}")
            return None
        
        # Determine provider from model name
        provider = "unknown"
        if "google" in model_name.lower():
            provider = "google"
        elif "meta" in model_name.lower() or "llama" in model_name.lower():
            provider = "meta-llama"
        elif "qwen" in model_name.lower():
            provider = "qwen"
        
        # Calculate overall LATAM score from category scores
        overall_scores = []
        for category_data in all_scores.values():
            if category_data and category_data.get("category_scores"):
                for cat_name, cat_data in category_data["category_scores"].items():
                    if cat_name.startswith("latam_"):
                        overall_scores.append(cat_data["score"])
        
        overall_latam_score = sum(overall_scores) / len(overall_scores) if overall_scores else None
        
        # Combine all scores
        combined_scores = {
            "model_name": model_name,
            "provider": provider,
            "categories": all_scores,
            "overall_latam_score": overall_latam_score,
            "tasks_mapping": tasks_mapping
        }
        
        return combined_scores
        
    except Exception as e:
        print(f"    Error processing {model_name}: {e}")
        return None

def parse_model_results(benchmark_outputs_dir: str, publish_dir: str) -> bool:
    """Parse model results and generate summary data for each model."""
    # Set up paths
    raw_data_dir = Path(benchmark_outputs_dir)
    publish_dir_path = Path(publish_dir)
    summaries_dir = publish_dir_path / "summaries"
    
    # Create summaries directory
    summaries_dir.mkdir(exist_ok=True)
    
    print(f"Processing model results from: {raw_data_dir}")
    
    if not raw_data_dir.exists():
        print("Raw data directory not found! Run copy_raw_data.py first.")
        return False
    
    # Process each model directory (skip subdirectories that start with provider names)
    model_dirs = []
    for d in raw_data_dir.iterdir():
        if d.is_dir() and d.name != "metadata.json":
            # Skip subdirectories that are copies of the provider__model format
            if "__" in d.name and any(provider in d.name for provider in ["google", "meta-llama", "qwen"]):
                continue
            model_dirs.append(d)
    
    if not model_dirs:
        print("No model directories found in raw_data!")
        return False
    
    print(f"Found {len(model_dirs)} model directories to process")
    
    all_summaries = {}
    
    for model_dir in model_dirs:
        print(f"\nProcessing {model_dir.name}...")
        try:
            summary = process_model_directory(model_dir)
            if summary:
                all_summaries[model_dir.name] = summary
                
                # Save individual model summary
                summary_file = summaries_dir / f"{model_dir.name}_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"  ✓ Summary saved to {summary_file}")
            else:
                print(f"  ✗ Failed to process {model_dir.name}")
                
        except Exception as e:
            print(f"  ✗ Error processing {model_dir.name}: {e}")
    
    # Save combined summaries
    combined_file = summaries_dir / "all_model_summaries.json"
    with open(combined_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n✓ Processing completed!")
    print(f"  Summaries directory: {summaries_dir}")
    print(f"  Models processed: {len(all_summaries)}")
    print(f"  Combined summaries: {combined_file}")
    
    return True

def main():
    """Main function for standalone execution."""
    import yaml
    config = load_config()
    benchmark_outputs = config["paths"]["benchmark_outputs"]
    publish_dir = config["paths"]["publish_dir"]
    return parse_model_results(benchmark_outputs, publish_dir)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
