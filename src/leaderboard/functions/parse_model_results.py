#!/usr/bin/env python3
"""
Script to parse raw model results and generate summary data for each model.
This processes the JSON results files and extracts hierarchical scores.
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

def extract_scores_from_results(results_data: Dict) -> Dict[str, Any]:
    """Extract and organize scores from the new benchmark results format."""
    
    # Extract individual task scores
    task_scores = {}
    for task_name, task_data in results_data.get("results", {}).items():
        if not task_name.startswith("latam_"):  # Skip top-level categories
            # Extract the main metric (look for "acc,all" or "acc,none" or similar)
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
            main_metric = None
            main_score = None
            main_stderr = None
            
            # Look for "acc,all" first (Portuguese format), then "acc,none" (Spanish format)
            for key, value in category_data.items():
                if key == "acc,all":
                    main_metric = "acc"
                    main_score = value
                    main_stderr = category_data.get("acc_stderr,all", 0.0)
                    break
                elif key == "acc,none":
                    main_metric = "acc"
                    main_score = value
                    main_stderr = category_data.get("acc_stderr,none", 0.0)
                    break
                elif key.endswith(",all") and not key.endswith("_stderr,all"):
                    metric_name = key.replace(",all", "")
                    if metric_name in ["acc", "exact_match", "acc_norm", "f1_macro"]:
                        main_metric = metric_name
                        main_score = value
                        stderr_key = f"{metric_name}_stderr,all"
                        main_stderr = category_data.get(stderr_key, 0.0)
                        break
                elif key.endswith(",none") and not key.endswith("_stderr,none"):
                    metric_name = key.replace(",none", "")
                    if metric_name in ["acc", "exact_match", "acc_norm", "f1_macro"]:
                        main_metric = metric_name
                        main_score = value
                        stderr_key = f"{metric_name}_stderr,none"
                        main_stderr = category_data.get(stderr_key, 0.0)
                        break
            
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

def process_model_directory(model_dir: Path) -> Dict[str, Any]:
    """Process both results.json and subdirectory results files in a model directory."""
    model_name = model_dir.name
    results_json = model_dir / "results.json"
    
    if not results_json.exists():
        print(f"  Warning: No results.json found in {model_name}")
        return None
    
    print(f"  Processing results.json and subdirectory files...")
    
    try:
        # Load tasks mapping
        tasks_mapping = load_tasks_mapping()
        
        # Process main results.json (Portuguese tasks)
        with open(results_json, 'r') as f:
            main_results_data = json.load(f)
        
        main_scores = extract_scores_from_results(main_results_data)
        main_scores["model_name"] = model_name
        
        # Process subdirectory results files (Spanish tasks)
        subdir_scores = {}
        # Look for results_*.json files in subdirectories recursively
        results_files = list(model_dir.rglob("results_*.json"))
        
        for results_file in results_files:
            try:
                with open(results_file, 'r') as f:
                    subdir_results_data = json.load(f)
                
                subdir_scores_data = extract_scores_from_results(subdir_results_data)
                subdir_scores_data["model_name"] = model_name
                
                # Determine language/category from the results
                if "latam_es" in subdir_results_data.get("results", {}):
                    subdir_scores["spanish"] = subdir_scores_data
                elif "latam_pr" in subdir_results_data.get("results", {}):
                    subdir_scores["portuguese"] = subdir_scores_data
                    
            except Exception as e:
                print(f"    Error processing {results_file.name}: {e}")
                continue
        
        # Determine provider from model name
        provider = "unknown"
        if "google" in model_name.lower():
            provider = "google"
        elif "meta" in model_name.lower() or "llama" in model_name.lower():
            provider = "meta-llama"
        elif "qwen" in model_name.lower():
            provider = "qwen"
        
        # Organize by categories
        all_scores = {}
        
        # Add Portuguese scores from main results.json
        if "latam_pr" in main_results_data.get("results", {}):
            all_scores["portuguese"] = main_scores
        
        # Add Spanish scores from subdirectory files
        if "spanish" in subdir_scores:
            all_scores["spanish"] = subdir_scores["spanish"]
        
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
        print(f"    Error processing {results_json.name}: {e}")
        return None

def parse_model_results(lm_eval_output_dir: str, publish_dir: str) -> bool:
    """Parse model results and generate summary data for each model."""
    # Set up paths
    raw_data_dir = Path(lm_eval_output_dir)
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
    lm_eval_output = config["paths"]["lm_eval_output"]
    publish_dir = config["paths"]["publish_dir"]
    return parse_model_results(lm_eval_output, publish_dir)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
