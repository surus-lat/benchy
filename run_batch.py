#!/usr/bin/env python3
"""Batch runner for multiple model evaluations."""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from zenml.logger import get_logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from main import main as run_single_model

logger = get_logger(__name__)

def setup_batch_logging():
    """Setup logging for batch runs."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create batch log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_file = log_dir / f"batch_run_{timestamp}.log"
    
    # Setup file handler for batch logs
    file_handler = logging.FileHandler(batch_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    batch_logger = logging.getLogger('benchy.batch')
    batch_logger.info(f"Batch logging started - log file: {batch_log_file}")
    
    return batch_log_file

def run_batch_evaluation():
    """Run evaluation for multiple models sequentially."""
    
    # Setup batch logging
    batch_log_file = setup_batch_logging()
    batch_logger = logging.getLogger('benchy.batch')
    
    # List of config files to run
    config_files = [
        "configs/model-1-qwen-4b-instruct-2507.yaml",
        "configs/model-2-1-gemma-e2b.yaml", 
        "configs/model-2-2-gemma-e4b.yaml",
        # Add more config files here
    ]
    
    # Track results
    results = []
    start_time = time.time()
    
    logger.info(f"Starting batch evaluation for {len(config_files)} models")
    batch_logger.info(f"=== BATCH EVALUATION STARTED ===")
    batch_logger.info(f"Total models: {len(config_files)}")
    batch_logger.info(f"Batch log file: {batch_log_file}")
    
    for i, config_file in enumerate(config_files, 1):
        batch_logger.info(f"Model {i}: {config_file}")
    
    for i, config_file in enumerate(config_files, 1):
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            batch_logger.error(f"Config file not found: {config_file}")
            continue
            
        logger.info(f"🚀 Running model {i}/{len(config_files)}: {config_file}")
        batch_logger.info(f"=== Starting Model {i}/{len(config_files)}: {config_file} ===")
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
            batch_logger.info(f"=== Model {i} COMPLETED SUCCESSFULLY in {model_time:.1f}s ===")
            
        except Exception as e:
            model_time = time.time() - model_start
            results.append({
                'config': config_file,
                'status': 'failed',
                'error': str(e),
                'duration': model_time
            })
            logger.error(f"❌ Model {i} failed after {model_time:.1f}s: {e}")
            batch_logger.error(f"=== Model {i} FAILED after {model_time:.1f}s ===")
            batch_logger.error(f"Error: {e}")
            
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
    
    # Log to batch logger
    batch_logger.info("=== BATCH EVALUATION SUMMARY ===")
    batch_logger.info(f"Total models: {len(config_files)}")
    batch_logger.info(f"Successful: {successful}")
    batch_logger.info(f"Failed: {failed}")
    batch_logger.info(f"Total time: {total_time:.1f}s")
    
    for result in results:
        status_emoji = "✅" if result['status'] == 'success' else "❌"
        config_name = Path(result['config']).stem
        logger.info(f"{status_emoji} {config_name}: {result['duration']:.1f}s")
        
        # Log to batch logger
        status_text = "SUCCESS" if result['status'] == 'success' else "FAILED"
        batch_logger.info(f"{config_name}: {status_text} in {result['duration']:.1f}s")
        if result['status'] == 'failed':
            batch_logger.error(f"  Error: {result.get('error', 'Unknown error')}")
    
    batch_logger.info(f"Batch log file: {batch_log_file}")
    batch_logger.info("=== BATCH EVALUATION COMPLETE ===")
    
    return results

if __name__ == "__main__":
    run_batch_evaluation()
