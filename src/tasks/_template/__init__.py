"""Template task module - copy this folder to create a new task.

To create a new task:
1. Copy this folder: cp -r src/tasks/_template src/tasks/my_task
2. Rename files and update class names
3. Implement the task logic in task.py
4. Update config: src/tasks/my_task/task.json (set runner_entrypoint + task_format)
5. Remove unused subtasks or swap in your datasets
6. Test: python eval.py --config configs/models/test.yaml --limit 5
"""

from .run import run_template_task

__all__ = ["run_template_task"]




