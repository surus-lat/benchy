#!/usr/bin/env python3
"""
Script to generate the final consolidated table from all model summaries.
This creates a table suitable for display on a website.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model_row(model_name: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a single row for the model in the final table."""
    
    row = {
        "model_name": model_name,
        "publisher": model_data.get("publisher", "unknown"),
        "full_model_name": model_data.get("full_model_name", model_name),
        "overall_latam_score": model_data.get("overall_latam_score"),
    }
    
    # Extract category scores
    categories = model_data.get("categories", {})
    
    # Portuguese scores
    if "portuguese" in categories and categories["portuguese"]:
        port_data = categories["portuguese"]
        row["portuguese_score"] = port_data.get("overall_score")
        
        # Individual Portuguese tasks
        task_scores = port_data.get("task_scores", {})
        for task_name, task_data in task_scores.items():
            row[f"portuguese_{task_name}"] = task_data.get("score")
    
    # Spanish scores
    if "spanish" in categories and categories["spanish"]:
        span_data = categories["spanish"]
        row["spanish_score"] = span_data.get("overall_score")
        
        # Individual Spanish tasks
        task_scores = span_data.get("task_scores", {})
        for task_name, task_data in task_scores.items():
            row[f"spanish_{task_name}"] = task_data.get("score")
        
        # Teleia subcategory
        teleia_score = task_scores.get("teleia")
        if teleia_score is None:
            teleia_score = span_data.get("category_scores", {}).get("teleia")
        if teleia_score:
            row["teleia_score"] = teleia_score.get("score")

            # Individual Teleia tasks
            teleia_tasks = ["teleia_cervantes_ave", "teleia_pce", "teleia_siele"]
            for task in teleia_tasks:
                if task in task_scores:
                    row[f"teleia_{task}"] = task_scores[task]["score"]
    
    return row

def generate_final_table(summaries_data: Dict[str, Any]) -> pd.DataFrame:
    """Generate the final consolidated table from all model summaries."""
    
    rows = []
    
    for model_name, model_data in summaries_data.items():
        try:
            row = create_model_row(model_name, model_data)
            rows.append(row)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    if not rows:
        print("No valid model data found!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by overall LATAM score (descending)
    if "overall_latam_score" in df.columns:
        df = df.sort_values("overall_latam_score", ascending=False, na_position='last')
    
    # Round numeric columns to 4 decimal places
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df[numeric_columns] = df[numeric_columns].round(4)
    
    return df

def create_detailed_breakdown(summaries_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a detailed breakdown of all scores for each model."""
    
    breakdown = {}
    
    for model_name, model_data in summaries_data.items():
        model_breakdown = {
            "model_name": model_name,
            "publisher": model_data.get("publisher", "unknown"),
            "full_model_name": model_data.get("full_model_name", model_name),
            "overall_latam_score": model_data.get("overall_latam_score"),
            "categories": {}
        }
        
        categories = model_data.get("categories", {})
        
        for category_name, category_data in categories.items():
            if not category_data:
                continue
                
            cat_breakdown = {
                "overall_score": category_data.get("overall_score"),
                "task_scores": category_data.get("task_scores", {}),
                "category_scores": category_data.get("category_scores", {}),
                "top_level_scores": category_data.get("top_level_scores", {})
            }
            
            model_breakdown["categories"][category_name] = cat_breakdown
        
        breakdown[model_name] = model_breakdown
    
    return breakdown

def main():
    """Main function to generate the final table."""
    config = load_config()
    
    # Set up paths
    leaderboard_dir = Path(config["paths"]["leaderboard_dir"])
    summaries_dir = leaderboard_dir / "summaries"
    publish_dir = leaderboard_dir / "publish"
    
    # Create publish directory
    publish_dir.mkdir(exist_ok=True)
    
    print(f"Generating final table from summaries in: {summaries_dir}")
    
    # Load all model summaries
    summaries_file = summaries_dir / "all_model_summaries.json"
    
    if not summaries_file.exists():
        print("Summaries file not found! Run parse_model_results.py first.")
        return
    
    with open(summaries_file, 'r') as f:
        summaries_data = json.load(f)
    
    print(f"Loaded summaries for {len(summaries_data)} models")
    
    # Generate final table
    print("\nGenerating final table...")
    final_table = generate_final_table(summaries_data)
    
    if final_table.empty:
        print("No data to generate table!")
        return
    
    # Save table in multiple formats
    print(f"Saving table with {len(final_table)} models and {len(final_table.columns)} columns")
    
    # CSV format
    csv_file = publish_dir / "leaderboard_table.csv"
    final_table.to_csv(csv_file, index=False)
    print(f"  ✓ CSV saved to: {csv_file}")
    
    # JSON format
    json_file = publish_dir / "leaderboard_table.json"
    final_table.to_json(json_file, orient="records", indent=2)
    print(f"  ✓ JSON saved to: {json_file}")
    
    # Excel format
    excel_file = publish_dir / "leaderboard_table.xlsx"
    final_table.to_excel(excel_file, index=False)
    print(f"  ✓ Excel saved to: {excel_file}")
    
    # Create detailed breakdown
    print("\nGenerating detailed breakdown...")
    detailed_breakdown = create_detailed_breakdown(summaries_data)
    
    breakdown_file = publish_dir / "detailed_breakdown.json"
    with open(breakdown_file, 'w') as f:
        json.dump(detailed_breakdown, f, indent=2)
    print(f"  ✓ Detailed breakdown saved to: {breakdown_file}")
    
    # Create summary statistics
    summary_stats = {
        "total_models": int(len(final_table)),
        "columns": list(final_table.columns),
        "numeric_columns": list(final_table.select_dtypes(include=['float64']).columns),
        "models_with_overall_score": int(final_table["overall_latam_score"].notna().sum()) if "overall_latam_score" in final_table.columns else 0,
        "top_5_models": final_table.head(5)[["model_name", "overall_latam_score"]].to_dict("records") if "overall_latam_score" in final_table.columns else []
    }
    
    stats_file = publish_dir / "summary_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  ✓ Summary statistics saved to: {stats_file}")
    
    # Display preview
    print(f"\n📊 Final Table Preview:")
    print(f"   Models: {len(final_table)}")
    print(f"   Columns: {len(final_table.columns)}")
    print(f"\n   Top 5 models by overall LATAM score:")
    if "overall_latam_score" in final_table.columns:
        top_5 = final_table.head(5)[["model_name", "overall_latam_score"]]
        for _, row in top_5.iterrows():
            score = row["overall_latam_score"]
            score_str = f"{score:.4f}" if pd.notna(score) else "N/A"
            print(f"     {row['model_name']}: {score_str}")
    
    print(f"\n✓ Final table generation completed!")
    print(f"  Publish directory: {publish_dir}")
    print(f"  Files created:")
    for file in publish_dir.glob("*"):
        print(f"    - {file.name}")

if __name__ == "__main__":
    main()
