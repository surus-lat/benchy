#!/usr/bin/env python3
"""Batch runner for multiple model evaluations."""

import os
import sys
import time
from pathlib import Path
from zenml.logger import get_logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from main import main as run_single_model

logger = get_logger(__name__)

def run_batch_evaluation():
    """Run evaluation for multiple models sequentially."""
    
    # List of config files to run
    config_files = [
        "configs/model-1-qwen.yaml",
        "configs/model-2-gemma.yaml", 
        "configs/model-3-llama.yaml",
        # Add more config files here
    ]
    
    # Track results
    results = []
    start_time = time.time()
    
    logger.info(f"Starting batch evaluation for {len(config_files)} models")
    
    for i, config_file in enumerate(config_files, 1):
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            continue
            
        logger.info(f"🚀 Running model {i}/{len(config_files)}: {config_file}")
        model_start = time.time()
        
        try:
            # Set the config file for main.py to use
            os.environ['BENCHY_CONFIG'] = config_file
            
            # Run the evaluation
            run_single_model()
            
            model_time = time.time() - model_start
            results.append({
                'config': config_file,
                'status': 'success',
                'duration': model_time
            })
            
            logger.info(f"✅ Model {i} completed in {model_time:.1f}s")
            
        except Exception as e:
            model_time = time.time() - model_start
            results.append({
                'config': config_file,
                'status': 'failed',
                'error': str(e),
                'duration': model_time
            })
            logger.error(f"❌ Model {i} failed after {model_time:.1f}s: {e}")
            
        finally:
            # Clean up environment variable
            if 'BENCHY_CONFIG' in os.environ:
                del os.environ['BENCHY_CONFIG']
    
    # Summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'failed'])
    
    logger.info("=" * 60)
    logger.info("📊 BATCH EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models: {len(config_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.1f}s")
    
    for result in results:
        status_emoji = "✅" if result['status'] == 'success' else "❌"
        config_name = Path(result['config']).stem
        logger.info(f"{status_emoji} {config_name}: {result['duration']:.1f}s")
        
    return results

if __name__ == "__main__":
    run_batch_evaluation()
