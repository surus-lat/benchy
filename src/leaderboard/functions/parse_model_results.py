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
  3. Creating a task config file in src/tasks/<task>/task.json
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
from config_loader import load_config

def load_task_config(task_name: str) -> Dict:
    """Load task-specific configuration from src/tasks/<task>/task.json."""
    project_root = Path(__file__).resolve().parents[3]
    tasks_root = project_root / "src" / "tasks"

    if not tasks_root.exists():
        return {}

    for config_path in tasks_root.rglob("task.json"):
        if config_path.parent.name == "_template":
            continue
        try:
            with open(config_path, "r") as f:
                task_config = json.load(f)
        except json.JSONDecodeError:
            continue
        if task_config.get("name") == task_name:
            return task_config

    return {}

def extract_model_info_from_config(model_dir: Path) -> Dict[str, str]:
    """Extract model information from run_config.yaml file.
    
    Supports the following fields in run_config.yaml:
    - model.name: Model name (required)
    - model.organization: Organization/publisher name (optional, overrides publisher extraction)
    - model.url: URL to model/organization (optional)
    
    If organization is not specified, publisher is extracted from model.name
    by splitting on "/" (e.g., "surus-ai/surus-extract" -> publisher: "surus-ai").
    """
    config_file = model_dir / "run_config.yaml"
    
    if not config_file.exists():
        print(f"    Warning: run_config.yaml not found in {model_dir}")
        return {
            "model_name": model_dir.name,
            "publisher": "unknown",
            "full_model_name": model_dir.name,
            "organization": None,
            "url": None
        }
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        model_config = config_data.get("model", {})
        
        # Extract model name from config
        full_model_name = model_config.get("name", model_dir.name)
        
        # Extract organization if explicitly provided
        organization = model_config.get("organization")
        
        # Extract URL if provided
        url = model_config.get("url")
        
        # Extract publisher and model name
        # Priority: 1) organization field, 2) split from model name, 3) unknown
        if organization:
            publisher = organization
            # If organization is set, model_name is just the name part
            if "/" in full_model_name:
                model_name = full_model_name.split("/")[1]
            else:
                model_name = full_model_name
        elif "/" in full_model_name:
            publisher = full_model_name.split("/")[0]
            model_name = full_model_name.split("/")[1]
        else:
            publisher = "unknown"
            model_name = full_model_name
        
        return {
            "model_name": model_name,
            "publisher": publisher,
            "full_model_name": full_model_name,
            "organization": organization,
            "url": url
        }
        
    except Exception as e:
        print(f"    Warning: Error reading run_config.yaml from {model_dir}: {e}")
        return {
            "model_name": model_dir.name,
            "publisher": "unknown",
            "full_model_name": model_dir.name,
            "organization": None,
            "url": None
        }

def extract_metric_from_task_data(task_data: Dict, task_name: str = None, normalize_tasks: list = None) -> tuple[Optional[str], Optional[float], Optional[float]]:
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
        elif key.endswith(",clean_translation") and not key.endswith("_stderr,clean_translation"):
            # Handle translation metrics like "chrf,clean_translation"
            metric_name = key.replace(",clean_translation", "")
            if metric_name in ["chrf", "bleu", "meteor"]:
                main_metric = metric_name
                # Normalize translation scores to 0-1 range if configured
                should_normalize = normalize_tasks and "translation" in normalize_tasks
                if should_normalize:
                    main_score = value / 100.0
                    stderr_key = f"{metric_name}_stderr,clean_translation"
                    main_stderr = task_data.get(stderr_key, 0.0) / 100.0  # Also normalize stderr
                else:
                    main_score = value
                    stderr_key = f"{metric_name}_stderr,clean_translation"
                    main_stderr = task_data.get(stderr_key, 0.0)
                break
    
    return main_metric, main_score, main_stderr

def extract_scores_from_results(results_data: Dict, task_name: str = None, normalize_tasks: list = None) -> Dict[str, Any]:
    """Extract and organize scores from the new benchmark results format."""
    
    # Extract individual task scores
    task_scores = {}
    for result_task_name, task_data in results_data.get("results", {}).items():
        if not result_task_name.startswith("latam_"):  # Skip top-level categories
            main_metric, main_score, main_stderr = extract_metric_from_task_data(task_data, task_name, normalize_tasks)
            
            if main_score is not None:
                task_scores[result_task_name] = {
                    "score": main_score,
                    "stderr": main_stderr,
                    "metric": main_metric,
                    "alias": result_task_name
                }
    
    # Extract category scores (latam_pr, latam_es, spanish, teleia, translation, etc.)
    category_scores = {}
    for category_name, category_data in results_data.get("results", {}).items():
        if category_name.startswith("latam_") or category_name in ["spanish", "teleia", "translation"]:
            main_metric, main_score, main_stderr = extract_metric_from_task_data(category_data, task_name, normalize_tasks)
            
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


def _load_latest_summary(task_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the most recent *_summary.json file from a task directory."""
    summary_files = sorted(task_dir.glob("*_summary.json"))
    if not summary_files:
        return None
    summary_file = summary_files[-1]
    try:
        with open(summary_file, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"    Warning: Failed to read summary {summary_file.name}: {exc}")
        return None


def _select_primary_metric(metrics: Dict[str, Any], preferred: Optional[str] = None) -> Tuple[Optional[str], Optional[float]]:
    """Pick a primary metric from a metrics dictionary."""
    if preferred and isinstance(metrics.get(preferred), (int, float)):
        return preferred, float(metrics[preferred])

    priority = [
        "acc",
        "exact_match",
        "f1_macro",
        "pearson",
        "comet",
        "chrf",
        "bleu",
        "overall_extraction_quality_score",
        "extraction_quality_score",
        "field_f1_partial",
        "mse",
    ]
    for key in priority:
        if isinstance(metrics.get(key), (int, float)):
            return key, float(metrics[key])

    return None, None


def _weighted_average(task_scores: Dict[str, Dict[str, Any]], per_task_metrics: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Compute a weighted average of task scores using valid_samples."""
    weighted_sum = 0.0
    total_weight = 0
    for task_name, score_data in task_scores.items():
        score = score_data.get("score")
        if not isinstance(score, (int, float)):
            continue
        weight = per_task_metrics.get(task_name, {}).get("valid_samples")
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        weighted_sum += float(score) * float(weight)
        total_weight += float(weight)
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def _scores_from_summary(
    summary_data: Dict[str, Any],
    category_key: Optional[str] = None,
    primary_metric: Optional[str] = None,
    subcategories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build score payload from summary JSON output."""
    per_subtask_metrics = summary_data.get("per_subtask_metrics", {}) or {}
    task_scores: Dict[str, Dict[str, Any]] = {}

    for subtask_name, metrics in per_subtask_metrics.items():
        metric_name, score = _select_primary_metric(metrics, primary_metric)
        if score is None:
            continue
        task_scores[subtask_name] = {
            "score": score,
            "stderr": 0.0,
            "metric": metric_name,
            "alias": subtask_name,
        }

    category_scores: Dict[str, Dict[str, Any]] = {}
    if category_key:
        category_score = None
        aggregated_metrics = summary_data.get("aggregated_metrics", {}) or {}
        if primary_metric and isinstance(aggregated_metrics.get(primary_metric), (int, float)):
            category_score = float(aggregated_metrics[primary_metric])
        else:
            category_score = _weighted_average(task_scores, per_subtask_metrics)

        if category_score is not None:
            category_scores[category_key] = {
                "score": category_score,
                "stderr": 0.0,
                "metric": primary_metric or "auto",
                "alias": category_key,
            }

    if subcategories:
        for subcategory in subcategories:
            sub_name = subcategory.get("name")
            filter_prefix = subcategory.get("filter_prefix", f"{sub_name}_")
            subtask_scores = {
                name: data
                for name, data in task_scores.items()
                if name.startswith(filter_prefix)
            }
            if not subtask_scores:
                continue
            sub_score = _weighted_average(subtask_scores, per_subtask_metrics)
            if sub_score is None:
                continue
            category_scores[sub_name] = {
                "score": sub_score,
                "stderr": 0.0,
                "metric": primary_metric or "auto",
                "alias": sub_name,
            }

    overall_score = None
    if category_scores:
        main_score = None
        if category_key and category_key in category_scores:
            main_score = category_scores[category_key]["score"]
        else:
            scores = [data["score"] for data in category_scores.values()]
            main_score = sum(scores) / len(scores) if scores else None
        overall_score = main_score

    return {
        "task_scores": task_scores,
        "category_scores": category_scores,
        "top_level_scores": category_scores,
        "overall_score": overall_score,
        "evaluation_time": summary_data.get("timestamp", "unknown"),
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

    summary_data = _load_latest_summary(spanish_dir)
    if summary_data:
        spanish_scores = _scores_from_summary(
            summary_data,
            category_key="latam_es",
            subcategories=[{"name": "teleia", "filter_prefix": "teleia_"}],
        )
        spanish_scores["model_name"] = model_name
        print(f"    ✓ Spanish results processed from summary file")
        return spanish_scores
    
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

    summary_data = _load_latest_summary(portuguese_dir)
    if summary_data:
        portuguese_scores = _scores_from_summary(
            summary_data,
            category_key="latam_pr",
        )
        portuguese_scores["model_name"] = model_name
        print(f"    ✓ Portuguese results processed from summary file")
        return portuguese_scores
    
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

def process_translation_results(model_dir: Path, model_name: str, normalize_tasks: list = None) -> Optional[Dict[str, Any]]:
    """Process Translation evaluation results for a model."""
    print(f"    Processing Translation results...")
    
    # Load task config to get output subdirectory
    task_config = load_task_config("translation")
    output_subdir = task_config.get("output", {}).get("subdirectory", "translation")
    
    # Look for results in the subdirectory
    task_dir = model_dir / output_subdir
    if not task_dir.exists():
        print(f"    Warning: Translation results directory not found: {task_dir}")
        return None

    summary_data = _load_latest_summary(task_dir)
    if summary_data:
        task_scores = _scores_from_summary(
            summary_data,
            category_key="translation",
            primary_metric="comet",
        )
        task_scores["model_name"] = model_name
        print(f"    ✓ Translation results processed from summary file")
        return task_scores
    
    # Look for results_*.json files in translation directory (including nested subdirectories)
    results_files = list(task_dir.rglob("results_*.json"))
    if not results_files:
        print(f"    Warning: No results_*.json files found in {task_dir}")
        return None
    
    # Process the first results file found
    results_file = results_files[0]
    try:
        with open(results_file, 'r') as f:
            task_results_data = json.load(f)
        
        task_scores = extract_scores_from_results(task_results_data, "translation", normalize_tasks)
        task_scores["model_name"] = model_name
        
        print(f"    ✓ Translation results processed from {results_file.relative_to(model_dir)}")
        return task_scores
        
    except Exception as e:
        print(f"    Error processing Translation results from {results_file.name}: {e}")
        return None

def standard_results_processor(model_dir: Path, model_name: str, task_config: Dict) -> Optional[Dict[str, Any]]:
    """Standard processor for tasks that follow the common results format."""
    task_name = task_config.get("name", "unknown")
    print(f"    Processing {task_name} results...")
    
    # Load task config to get output subdirectory
    task_config_file = load_task_config(task_name)
    output_subdir = task_config_file.get("output", {}).get("subdirectory", task_name)
    
    # Look for results in the subdirectory
    task_dir = model_dir / output_subdir
    if not task_dir.exists():
        print(f"    Warning: {task_name} results directory not found: {task_dir}")
        return None

    summary_data = _load_latest_summary(task_dir)
    if summary_data:
        category_key = task_config.get("category_score_key")
        subcategories = task_config.get("subcategories", [])
        task_scores = _scores_from_summary(
            summary_data,
            category_key=category_key,
            subcategories=subcategories,
        )
        task_scores["model_name"] = model_name
        print(f"    ✓ {task_name} results processed from summary file")
        return task_scores
    
    # Look for results_*.json files
    results_files = list(task_dir.rglob("results_*.json"))
    if not results_files:
        print(f"    Warning: No results_*.json files found in {task_dir}")
        return None
    
    # Process the first results file found
    results_file = results_files[0]
    try:
        with open(results_file, 'r') as f:
            task_results_data = json.load(f)
        
        task_scores = extract_scores_from_results(task_results_data, task_name, None)
        task_scores["model_name"] = model_name
        
        print(f"    ✓ {task_name} results processed from {results_file.relative_to(model_dir)}")
        return task_scores
        
    except Exception as e:
        print(f"    Error processing {task_name} results from {results_file.name}: {e}")
        return None

def portuguese_results_processor(model_dir: Path, model_name: str, task_config: Dict) -> Optional[Dict[str, Any]]:
    """Specialized processor for Portuguese tasks that use results.json format."""
    task_name = task_config.get("name", "portuguese")
    print(f"    Processing {task_name} results...")
    
    # Load task config to get output subdirectory
    task_config_file = load_task_config(task_name)
    output_subdir = task_config_file.get("output", {}).get("subdirectory", task_name)
    
    # Look for results in the subdirectory
    task_dir = model_dir / output_subdir
    if not task_dir.exists():
        print(f"    Warning: {task_name} results directory not found: {task_dir}")
        return None

    summary_data = _load_latest_summary(task_dir)
    if summary_data:
        category_key = task_config.get("category_score_key")
        task_scores = _scores_from_summary(
            summary_data,
            category_key=category_key,
        )
        task_scores["model_name"] = model_name
        print(f"    ✓ {task_name} results processed from summary file")
        return task_scores
    
    # Look for results.json (Portuguese uses different file pattern)
    results_file = task_dir / "results.json"
    if not results_file.exists():
        print(f"    Warning: No results.json found in {task_dir}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            task_results_data = json.load(f)
        
        task_scores = extract_scores_from_results(task_results_data, task_name, None)
        task_scores["model_name"] = model_name
        
        print(f"    ✓ {task_name} results processed from {results_file.name}")
        return task_scores
        
    except Exception as e:
        print(f"    Error processing {task_name} results from {results_file.name}: {e}")
        return None

def translation_results_processor(model_dir: Path, model_name: str, task_config: Dict) -> Optional[Dict[str, Any]]:
    """Specialized processor for translation tasks with metric selection."""
    task_name = task_config.get("name", "translation")
    primary_metric = task_config.get("primary_metric", "chrf")
    normalize_tasks = task_config.get("normalize_tasks", ["translation"])
    
    print(f"    Processing {task_name} results...")
    
    # Load task config to get output subdirectory
    task_config_file = load_task_config(task_name)
    output_subdir = task_config_file.get("output", {}).get("subdirectory", task_name)
    
    # Look for results in the subdirectory
    task_dir = model_dir / output_subdir
    if not task_dir.exists():
        print(f"    Warning: {task_name} results directory not found: {task_dir}")
        return None

    summary_data = _load_latest_summary(task_dir)
    if summary_data:
        category_key = task_config.get("category_score_key")
        task_scores = _scores_from_summary(
            summary_data,
            category_key=category_key,
            primary_metric=primary_metric,
        )
        task_scores["model_name"] = model_name
        print(f"    ✓ {task_name} results processed from summary file")
        return task_scores
    
    # Look for results_*.json files
    results_files = list(task_dir.rglob("results_*.json"))
    if not results_files:
        print(f"    Warning: No results_*.json files found in {task_dir}")
        return None
    
    # Process the first results file found
    results_file = results_files[0]
    try:
        with open(results_file, 'r') as f:
            task_results_data = json.load(f)
        
        task_scores = extract_scores_from_results(task_results_data, task_name, normalize_tasks)
        task_scores["model_name"] = model_name
        
        print(f"    ✓ {task_name} results processed from {results_file.relative_to(model_dir)}")
        return task_scores
        
    except Exception as e:
        print(f"    Error processing {task_name} results from {results_file.name}: {e}")
        return None

def structured_extraction_results_processor(model_dir: Path, model_name: str, task_config: Dict) -> Optional[Dict[str, Any]]:
    """Specialized processor for structured extraction tasks."""
    import shutil
    
    task_name = task_config.get("name", "structured_extraction")
    print(f"    Processing {task_name} results...")
    
    # Load task config to get output subdirectory
    task_config_file = load_task_config(task_name)
    output_subdir = task_config_file.get("output", {}).get("subdirectory", task_name)
    
    # Look for results in the subdirectory
    task_dir = model_dir / output_subdir
    if not task_dir.exists():
        print(f"    Warning: {task_name} results directory not found: {task_dir}")
        return None

    summary_data = _load_latest_summary(task_dir)
    if summary_data:
        aggregated_metrics = summary_data.get("aggregated_metrics", {}) or {}
        per_subtask_metrics = summary_data.get("per_subtask_metrics", {}) or {}

        eqs = aggregated_metrics.get("overall_extraction_quality_score")
        if not isinstance(eqs, (int, float)):
            eqs = aggregated_metrics.get("extraction_quality_score", 0.0)

        composite_scores = []
        weights = []
        for subtask_name, metrics in per_subtask_metrics.items():
            stats = metrics.get("composite_score_stats", {})
            mean_score = stats.get("mean")
            if isinstance(mean_score, (int, float)):
                weight = metrics.get("valid_samples", 0)
                if isinstance(weight, (int, float)) and weight > 0:
                    composite_scores.append(float(mean_score) * float(weight))
                    weights.append(float(weight))

        composite_mean = sum(composite_scores) / sum(weights) if weights else 0.0
        schema_validity = aggregated_metrics.get("schema_validity_rate", 0.0)
        f1_partial = aggregated_metrics.get("field_f1_partial", 0.0)
        hallucination_rate = aggregated_metrics.get("hallucination_rate", 0.0)

        task_scores = {
            "extraction_quality_score": {
                "score": float(eqs) if isinstance(eqs, (int, float)) else 0.0,
                "stderr": 0.0,
                "metric": "eqs",
                "alias": "extraction_quality_score",
            },
            "composite_score": {
                "score": composite_mean,
                "stderr": 0.0,
                "metric": "composite_mean",
                "alias": "composite_score",
            },
            "schema_validity": {
                "score": schema_validity,
                "stderr": 0.0,
                "metric": "validity_rate",
                "alias": "schema_validity",
            },
            "field_f1_partial": {
                "score": f1_partial,
                "stderr": 0.0,
                "metric": "f1",
                "alias": "field_f1_partial",
            },
            "hallucination_rate": {
                "score": hallucination_rate,
                "stderr": 0.0,
                "metric": "hallucination_rate",
                "alias": "hallucination_rate",
            },
        }

        category_key = task_config.get("category_score_key", task_name)
        category_scores = {
            category_key: {
                "score": float(eqs) if isinstance(eqs, (int, float)) else 0.0,
                "stderr": 0.0,
                "metric": "eqs",
                "alias": category_key,
            }
        }

        report_files = list(task_dir.glob("*_summary.txt"))
        if not report_files:
            report_files = list(task_dir.rglob("*_report.txt"))
        if report_files:
            report_file = report_files[0]
            config = load_config()
            publish_dir = Path(config["paths"]["publish_dir"])
            summaries_dir = publish_dir / "summaries"
            summaries_dir.mkdir(parents=True, exist_ok=True)

            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            report_dest = summaries_dir / f"{safe_model_name}_structured_extraction_report.txt"
            shutil.copy2(report_file, report_dest)
            print(f"    ✓ Report copied to {report_dest.name}")

        result = {
            "model_name": model_name,
            "task_scores": task_scores,
            "category_scores": category_scores,
            "top_level_scores": category_scores,
            "overall_score": float(eqs) if isinstance(eqs, (int, float)) else 0.0,
            "evaluation_time": summary_data.get("timestamp", "unknown"),
        }

        print(f"    ✓ {task_name} results processed from summary file")
        return result
    
    # Look for metrics JSON files (structured extraction format)
    metrics_files = list(task_dir.glob("*_metrics.json"))
    if not metrics_files:
        print(f"    Warning: No *_metrics.json files found in {task_dir}")
        return None
    
    # Process the first metrics file found
    metrics_file = metrics_files[0]
    try:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        # Extract metrics from structured extraction format
        metrics = metrics_data.get("metrics", {})
        
        # Get EQS as main metric for leaderboard (better than composite score)
        # Use overall_extraction_quality_score if available (accounts for invalid responses)
        # Fall back to extraction_quality_score for backward compatibility
        eqs = metrics.get("overall_extraction_quality_score", metrics.get("extraction_quality_score", 0.0))
        
        # Get composite score stats for reference
        composite_stats = metrics.get("composite_score_stats", {})
        composite_mean = composite_stats.get("mean", 0.0)
        composite_stdev = composite_stats.get("stdev", 0.0)
        
        # Also get other key metrics
        schema_validity = metrics.get("schema_validity_rate", 0.0)
        f1_partial = metrics.get("field_f1_partial", 0.0)
        hallucination_rate = metrics.get("hallucination_rate", 0.0)
        
        # Copy report.txt file to summaries if it exists
        report_files = list(task_dir.glob("*_report.txt"))
        if report_files:
            report_file = report_files[0]
            # Get publish directory from config
            config = load_config()
            publish_dir = Path(config["paths"]["publish_dir"])
            summaries_dir = publish_dir / "summaries"
            summaries_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy report with model name (sanitize model name for filesystem)
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            report_dest = summaries_dir / f"{safe_model_name}_structured_extraction_report.txt"
            shutil.copy2(report_file, report_dest)
            print(f"    ✓ Report copied to {report_dest.name}")
        
        # Build task scores (individual metrics)
        task_scores = {
            "extraction_quality_score": {
                "score": eqs,
                "stderr": 0.0,
                "metric": "eqs",
                "alias": "extraction_quality_score"
            },
            "composite_score": {
                "score": composite_mean,
                "stderr": composite_stdev,
                "metric": "composite_mean",
                "alias": "composite_score"
            },
            "schema_validity": {
                "score": schema_validity,
                "stderr": 0.0,
                "metric": "validity_rate",
                "alias": "schema_validity"
            },
            "field_f1_partial": {
                "score": f1_partial,
                "stderr": 0.0,
                "metric": "f1",
                "alias": "field_f1_partial"
            },
            "hallucination_rate": {
                "score": hallucination_rate,
                "stderr": 0.0,
                "metric": "hallucination_rate",
                "alias": "hallucination_rate"
            }
        }
        
        # Build category scores (use EQS as main category score)
        # Use category_score_key from task config, fallback to task_name
        category_key = task_config.get("category_score_key", task_name)
        category_scores = {
            category_key: {
                "score": eqs,
                "stderr": 0.0,  # EQS doesn't have stderr in current format
                "metric": "eqs",
                "alias": category_key
            }
        }
        
        result = {
            "model_name": model_name,
            "task_scores": task_scores,
            "category_scores": category_scores,
            "top_level_scores": category_scores,
            "overall_score": eqs,  # Use EQS as overall score
            "evaluation_time": metrics_data.get("timestamp", "unknown")
        }
        
        print(f"    ✓ {task_name} results processed from {metrics_file.name}")
        print(f"    ✓ EQS (Primary): {eqs:.4f}")
        print(f"    ✓ EQS Components: Validity={schema_validity:.2%}, F1={f1_partial:.4f}, Hallucination={hallucination_rate:.2%}")
        print(f"    ✓ Composite Score: {composite_mean:.4f} ± {composite_stdev:.4f}")
        
        return result
        
    except Exception as e:
        print(f"    Error processing {task_name} results from {metrics_file.name}: {e}")
        return None

def get_task_processor(processor_type: str) -> callable:
    """Get the appropriate processor function based on type."""
    processors = {
        "standard_results_processor": standard_results_processor,
        "portuguese_results_processor": portuguese_results_processor,
        "translation_results_processor": translation_results_processor,
        "structured_extraction_results_processor": structured_extraction_results_processor,
    }
    return processors.get(processor_type, standard_results_processor)

def get_available_task_processors() -> Dict[str, callable]:
    """Get dictionary of available task processors (legacy compatibility)."""
    return {
        "spanish": process_spanish_results,
        "portuguese": process_portuguese_results,
        "translation": process_translation_results,
        # Add new tasks here as they are implemented
    }

def process_model_directory(model_dir: Path, config: Dict = None) -> Dict[str, Any]:
    """Process model results using modular task system."""
    model_name = model_dir.name
    print(f"  Processing {model_name}...")
    
    try:
        # Load configuration if not provided
        if config is None:
            config = load_config()
        
        # Extract model information from config file
        model_info = extract_model_info_from_config(model_dir)
        config_model_name = model_info["model_name"]
        publisher = model_info["publisher"]
        full_model_name = model_info["full_model_name"]
        organization = model_info.get("organization")
        url = model_info.get("url")
        
        print(f"    Model info from config: {config_model_name} (publisher: {publisher})")
        if organization:
            print(f"      Organization: {organization}")
        if url:
            print(f"      URL: {url}")
        
        # Get leaderboard configuration
        leaderboard_config = config.get("leaderboard", {})
        main_categories = leaderboard_config.get("overall_score_categories", ["latam_es", "latam_pr", "translation"])
        normalize_tasks = leaderboard_config.get("normalize_scores", ["translation"])
        task_definitions = leaderboard_config.get("tasks", {})
        
        # Process each task using modular system
        all_scores = {}
        
        for task_name, task_config in task_definitions.items():
            # Add task name to config for processors
            task_config_with_name = task_config.copy()
            task_config_with_name["name"] = task_name
            task_config_with_name["normalize_tasks"] = normalize_tasks
            
            # Get the appropriate processor
            processor_type = task_config.get("processor", "standard_results_processor")
            processor_func = get_task_processor(processor_type)
            
            # Process the task
            task_scores = processor_func(model_dir, config_model_name, task_config_with_name)
            if task_scores:
                all_scores[task_name] = task_scores
        
        if not all_scores:
            print(f"  Warning: No valid results found for {model_name}")
            return None
        
        # Generate overall translation score if none exists
        if "translation" in all_scores:
            translation_data = all_scores["translation"]
            if not translation_data.get("category_scores") or "translation" not in translation_data["category_scores"]:
                # Calculate average of all translation task scores
                task_scores = translation_data.get("task_scores", {})
                if task_scores:
                    translation_scores = [task_data["score"] for task_data in task_scores.values()]
                    avg_translation_score = sum(translation_scores) / len(translation_scores)
                    
                    # The individual task scores are already normalized, so the average is already in 0-1 range
                    final_translation_score = avg_translation_score
                    print(f"    ✓ Generated overall translation score: {final_translation_score:.4f}")
                    
                    # Add the calculated translation score to category_scores
                    if "category_scores" not in translation_data:
                        translation_data["category_scores"] = {}
                    
                    translation_data["category_scores"]["translation"] = {
                        "score": final_translation_score,
                        "stderr": 0.0,  # We don't have stderr for the average
                        "metric": "chrf",  # Assuming CHRF as the main metric
                        "alias": "translation"
                    }

        # Combine all scores
        combined_scores = {
            "model_name": config_model_name,
            "publisher": publisher,
            "full_model_name": full_model_name,
            "categories": all_scores
        }
        
        # Add organization and url if available
        if organization:
            combined_scores["organization"] = organization
        if url:
            combined_scores["url"] = url
        
        return combined_scores
        
    except Exception as e:
        print(f"    Error processing {model_name}: {e}")
        return None

def parse_model_results(benchmark_outputs_dir: str, publish_dir: str) -> bool:
    """Parse model results and generate summary data for each model."""
    # Load configuration
    config = load_config()
    
    # Set up paths
    raw_data_dir = Path(benchmark_outputs_dir)
    publish_dir_path = Path(publish_dir)
    summaries_dir = publish_dir_path / "summaries"
    
    # Create summaries directory
    publish_dir_path.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
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
            summary = process_model_directory(model_dir, config)
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
