"""Logging utilities for Benchy - file and console logging setup."""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from zenml.logger import get_logger as get_zenml_logger


class BenchyLoggingSetup:
    """Configure comprehensive logging for Benchy runs."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate log file name with timestamp and model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get('model', {}).get('name', 'unknown_model')
        # Clean model name for filename
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        
        self.log_filename = f"benchy_{safe_model_name}_{timestamp}.log"
        self.log_filepath = self.log_dir / self.log_filename
        
        # Setup both Python logging and ZenML logging
        self.setup_python_logging()
        self.zenml_logger = get_zenml_logger(__name__)
        
    def setup_python_logging(self):
        """Setup Python standard logging to both file and console."""
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Add our handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Log the setup
        logger = logging.getLogger('benchy.logging')
        logger.info(f"Logging initialized - log file: {self.log_filepath}")
        logger.info(f"Model: {self.config.get('model', {}).get('name', 'unknown')}")
        logger.info(f"Tasks: {self.config.get('evaluation', {}).get('tasks', 'unknown')}")
        
    def log_config(self):
        """Log the complete configuration."""
        logger = logging.getLogger('benchy.config')
        logger.info("=== Configuration ===")
        
        # Model config
        model_config = self.config.get('model', {})
        logger.info(f"Model Name: {model_config.get('name', 'N/A')}")
        logger.info(f"Model dtype: {model_config.get('dtype', 'N/A')}")
        logger.info(f"Model max_length: {model_config.get('max_length', 'N/A')}")
        
        # Evaluation config
        eval_config = self.config.get('evaluation', {})
        logger.info(f"Tasks: {eval_config.get('tasks', 'N/A')}")
        logger.info(f"Device: {eval_config.get('device', 'N/A')}")
        logger.info(f"Batch size: {eval_config.get('batch_size', 'N/A')}")
        logger.info(f"Output path: {eval_config.get('output_path', 'N/A')}")
        logger.info(f"Log samples: {eval_config.get('log_samples', 'N/A')}")
        if 'limit' in eval_config:
            logger.info(f"Limit: {eval_config['limit']} (testing mode)")
        
        # Paths
        venv_config = self.config.get('venvs', {})
        logger.info(f"LM Eval path: {venv_config.get('lm_eval', 'N/A')}")
        logger.info(f"Leaderboard path: {venv_config.get('leaderboard', 'N/A')}")
        
        logger.info("=== End Configuration ===")
    
    def log_command(self, command: str, step_name: str = "command"):
        """Log the command being executed."""
        logger = logging.getLogger(f'benchy.{step_name}')
        logger.info(f"Executing: {command}")
    
    def log_step_start(self, step_name: str, **kwargs):
        """Log the start of a pipeline step."""
        logger = logging.getLogger(f'benchy.{step_name}')
        logger.info(f"=== Starting {step_name} ===")
        for key, value in kwargs.items():
            logger.info(f"{key}: {value}")
    
    def log_step_end(self, step_name: str, success: bool = True, **kwargs):
        """Log the end of a pipeline step."""
        logger = logging.getLogger(f'benchy.{step_name}')
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"=== {step_name} {status} ===")
        for key, value in kwargs.items():
            logger.info(f"{key}: {value}")
    
    def log_subprocess_output(self, line: str, step_name: str = "subprocess"):
        """Log subprocess output with proper formatting."""
        logger = logging.getLogger(f'benchy.{step_name}')
        # Remove any existing prefixes to avoid double-prefixing
        clean_line = line.strip()
        if clean_line:
            logger.info(f"[{step_name}] {clean_line}")
    
    def get_log_filepath(self) -> Path:
        """Get the current log file path."""
        return self.log_filepath
    
    def log_summary(self, results):
        """Log a summary of the run results."""
        logger = logging.getLogger('benchy.summary')
        logger.info("=== RUN SUMMARY ===")
        
        # Handle both dictionary and ZenML response objects
        if hasattr(results, 'get'):
            # Dictionary-like object
            model_name = results.get('model_name', 'unknown')
            return_code = results.get('return_code', 'unknown')
            error = results.get('error')
        elif hasattr(results, 'body') and hasattr(results.body, 'status'):
            # ZenML PipelineRunResponse object
            model_name = self.config.get('model', {}).get('name', 'unknown')
            status = results.body.status.value if hasattr(results.body.status, 'value') else str(results.body.status)
            return_code = 0 if status == 'completed' else 1
            error = None
        else:
            # Fallback
            model_name = self.config.get('model', {}).get('name', 'unknown')
            return_code = 'unknown'
            error = None
            
        logger.info(f"Model: {model_name}")
        logger.info(f"Return code: {return_code}")
        logger.info(f"Log file: {self.log_filepath}")
        
        # Log any errors
        if return_code != 0:
            logger.error("Run failed - check logs above for details")
            if error:
                logger.error(f"Error: {error}")
        else:
            logger.info("Run completed successfully")
            
        logger.info("=== END SUMMARY ===")


def setup_file_logging(config: Dict[str, Any], log_dir: str = "logs") -> BenchyLoggingSetup:
    """Setup file logging for a Benchy run."""
    return BenchyLoggingSetup(config, log_dir)
