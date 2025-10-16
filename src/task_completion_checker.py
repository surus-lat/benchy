"""Task completion checking functionality for resuming failed runs."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class TaskCompletionChecker:
    """Checks if tasks have already completed successfully by looking for done files."""
    
    def __init__(self, output_path: str, run_id: str, model_name: str):
        """
        Initialize the task completion checker.
        
        Args:
            output_path: Base output path for results
            run_id: Run ID for this execution
            model_name: The model being evaluated
        """
        self.output_path = output_path
        self.run_id = run_id
        self.model_name = model_name
        self.model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
        
    def is_task_completed(self, task_name: str) -> bool:
        """
        Check if a task has already completed successfully by looking for a done file.
        
        Args:
            task_name: Name of the task to check (e.g., 'spanish', 'portuguese', 'translation', 'structured_extraction')
            
        Returns:
            True if task appears to be completed, False otherwise
        """
        task_output_dir = f"{self.model_output_path}/{task_name}"
        done_file = f"{task_output_dir}/.done"
        
        if os.path.exists(done_file):
            logger.info(f"Task {task_name} already completed (found {done_file})")
            return True
        else:
            logger.info(f"Task {task_name} not completed (no {done_file} found)")
            return False
    
    def get_completed_tasks(self, requested_tasks: List[str]) -> Dict[str, bool]:
        """
        Check completion status for multiple tasks.
        
        Args:
            requested_tasks: List of task names to check
            
        Returns:
            Dictionary mapping task names to completion status
        """
        completion_status = {}
        for task in requested_tasks:
            completion_status[task] = self.is_task_completed(task)
            
        completed_tasks = [task for task, completed in completion_status.items() if completed]
        if completed_tasks:
            logger.info(f"Found completed tasks: {completed_tasks}")
        else:
            logger.info("No completed tasks found")
            
        return completion_status
    
    def log_completion_summary(self, completion_status: Dict[str, bool]) -> None:
        """Log a summary of task completion status."""
        completed = [task for task, status in completion_status.items() if status]
        pending = [task for task, status in completion_status.items() if not status]
        
        logger.info("=" * 60)
        logger.info("TASK COMPLETION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output Path: {self.model_output_path}")
        logger.info("")
        
        if completed:
            logger.info(f"✅ COMPLETED TASKS ({len(completed)}):")
            for task in completed:
                logger.info(f"  - {task}")
        else:
            logger.info("✅ COMPLETED TASKS: None")
            
        if pending:
            logger.info(f"⏳ PENDING TASKS ({len(pending)}):")
            for task in pending:
                logger.info(f"  - {task}")
        else:
            logger.info("⏳ PENDING TASKS: None")
            
        logger.info("=" * 60)


def write_task_done_file(task_output_path: str) -> None:
    """
    Write a done file to mark task completion.
    
    Args:
        task_output_path: Path to the task output directory
    """
    # Ensure directory exists
    Path(task_output_path).mkdir(parents=True, exist_ok=True)
    
    done_file = f"{task_output_path}/.done"
    with open(done_file, 'w') as f:
        f.write("")  # Empty file
    logger.info(f"Task completion marked: {done_file}")
