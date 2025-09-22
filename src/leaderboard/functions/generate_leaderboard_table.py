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

def create_leaderboard_table(summaries: List[Dict]) -> List[Dict]:
    """Generate the leaderboard table in the expected format."""
    leaderboard_data = []
    
    for model_data in summaries:
        model_name = model_data["model_name"]
        provider = model_data["provider"]
        categories = model_data.get("categories", {})
        overall_latam_score = model_data.get("overall_latam_score")
        
        # Initialize the row with basic info
        row = {
            "model_name": model_name,
            "provider": provider,
            "overall_latam_score": round(overall_latam_score, 4) if overall_latam_score is not None else None
        }
        
        # Process Portuguese scores
        if "portuguese" in categories:
            portuguese_data = categories["portuguese"]
            
            # Get Portuguese category score (latam_pr)
            portuguese_score = None
            if "latam_pr" in portuguese_data.get("category_scores", {}):
                portuguese_score = portuguese_data["category_scores"]["latam_pr"]["score"]
                row["portuguese_score"] = round(portuguese_score, 4)
            
            # Add individual Portuguese task scores
            for task_name, task_data in portuguese_data.get("task_scores", {}).items():
                row[f"portuguese_{task_name}"] = round(task_data["score"], 4)
        
        # Process Spanish scores
        if "spanish" in categories:
            spanish_data = categories["spanish"]
            
            # Get Spanish category score (latam_es)
            spanish_score = None
            if "latam_es" in spanish_data.get("category_scores", {}):
                spanish_score = spanish_data["category_scores"]["latam_es"]["score"]
                row["spanish_score"] = round(spanish_score, 4)
            
            # Add individual Spanish task scores
            for task_name, task_data in spanish_data.get("task_scores", {}).items():
                row[f"spanish_{task_name}"] = round(task_data["score"], 4)
            
            # Add Teleia subcategory scores if they exist
            if "teleia" in spanish_data.get("category_scores", {}):
                teleia_score = spanish_data["category_scores"]["teleia"]["score"]
                row["teleia_score"] = round(teleia_score, 4)
                
                # Add individual Teleia task scores
                for task_name, task_data in spanish_data.get("task_scores", {}).items():
                    if task_name.startswith("teleia_"):
                        row[f"teleia_{task_name}"] = round(task_data["score"], 4)
        
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
    
    # Generate leaderboard table
    leaderboard_data = create_leaderboard_table(summaries)
    
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
