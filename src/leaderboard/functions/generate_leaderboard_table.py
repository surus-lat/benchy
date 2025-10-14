#!/usr/bin/env python3
"""
Generate leaderboard table in the format expected by the frontend.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_all_summaries(summaries_dir: Path) -> List[Dict]:
    """Load all model summaries."""
    all_summaries_file = summaries_dir / "all_model_summaries.json"
    
    if not all_summaries_file.exists():
        print("No combined summaries found! Run parse_model_results.py first.")
        return []
    
    with open(all_summaries_file, 'r') as f:
        data = json.load(f)
    
    # Convert dictionary to list of model data
    if isinstance(data, dict):
        return list(data.values())
    else:
        return data

def process_task_scores(row: Dict, task_name: str, task_data: Dict, task_config: Dict) -> None:
    """Process scores for a specific task based on its configuration."""
    output_prefix = task_config.get("output_prefix", task_name)
    category_score_key = task_config.get("category_score_key")
    exclude_tasks = task_config.get("exclude_tasks", [])
    
    # Add category score
    if category_score_key and category_score_key in task_data.get("category_scores", {}):
        category_score = task_data["category_scores"][category_score_key]["score"]
        row[f"{output_prefix}_score"] = round(category_score, 4)
    
    # Add individual task scores
    for individual_task_name, individual_task_data in task_data.get("task_scores", {}).items():
        if individual_task_name not in exclude_tasks:
            row[f"{output_prefix}_{individual_task_name}"] = round(individual_task_data["score"], 4)
    
    # Process subcategories if defined
    subcategories = task_config.get("subcategories", [])
    for subcategory in subcategories:
        subcategory_name = subcategory["name"]
        subcategory_prefix = subcategory["prefix"]
        filter_prefix = subcategory.get("filter_prefix", f"{subcategory_name}_")
        
        # Add subcategory score if it exists
        if subcategory_name in task_data.get("category_scores", {}):
            subcategory_score = task_data["category_scores"][subcategory_name]["score"]
            row[f"{subcategory_prefix}_score"] = round(subcategory_score, 4)
        
        # Add individual subcategory task scores
        for individual_task_name, individual_task_data in task_data.get("task_scores", {}).items():
            if individual_task_name.startswith(filter_prefix):
                row[f"{subcategory_prefix}_{individual_task_name}"] = round(individual_task_data["score"], 4)

def create_leaderboard_table(summaries: List[Dict], config: Dict = None) -> List[Dict]:
    """Generate the leaderboard table in the expected format using modular task system."""
    if config is None:
        from .parse_model_results import load_config
        config = load_config()
    
    leaderboard_config = config.get("leaderboard", {})
    task_definitions = leaderboard_config.get("tasks", {})
    
    leaderboard_data = []
    
    for model_data in summaries:
        model_name = model_data["model_name"]
        publisher = model_data.get("publisher", "unknown")
        full_model_name = model_data.get("full_model_name", model_name)
        categories = model_data.get("categories", {})
        overall_latam_score = model_data.get("overall_latam_score")
        
        # Initialize the row with basic info
        row = {
            "model_name": model_name,
            "publisher": publisher,
            "full_model_name": full_model_name,
            "overall_latam_score": round(overall_latam_score, 4) if overall_latam_score is not None else None
        }
        
        # Process each task using modular system
        for task_name, task_config in task_definitions.items():
            if task_name in categories:
                process_task_scores(row, task_name, categories[task_name], task_config)
        
        leaderboard_data.append(row)
    
    return leaderboard_data

def generate_leaderboard_table(publish_dir: str) -> bool:
    """Generate the leaderboard table."""
    # Get paths
    publish_dir_path = Path(publish_dir)
    summaries_dir = publish_dir_path / "summaries"
    
    # Create publish directory if it doesn't exist
    publish_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating leaderboard table from: {summaries_dir}")
    
    # Load all summaries
    summaries = load_all_summaries(summaries_dir)
    
    if not summaries:
        print("No summaries found!")
        return False
    
    print(f"Found {len(summaries)} model summaries")
    
    # Load configuration
    from .parse_model_results import load_config
    config = load_config()
    
    # Generate leaderboard table
    leaderboard_data = create_leaderboard_table(summaries, config)
    
    # Save as JSON
    output_file = publish_dir_path / "leaderboard_table.json"
    with open(output_file, 'w') as f:
        json.dump(leaderboard_data, f, indent=2)
    
    print(f"✓ Leaderboard table saved to: {output_file}")
    print(f"  Models: {len(leaderboard_data)}")
    
    # Also save as CSV for easy viewing
    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)
        csv_file = publish_dir_path / "leaderboard_table.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ CSV version saved to: {csv_file}")
    
    return True

def main():
    """Main function for standalone execution."""
    import yaml
    config = load_config()
    publish_dir = config["paths"]["publish_dir"]
    return generate_leaderboard_table(publish_dir)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
