# Task Template - Handler System

This template demonstrates how to create new benchmark tasks using the handler system.

## Quick Start

1. **Copy this template folder** and rename it:
   ```bash
   cp -r src/tasks/_template_handler src/tasks/my_task
   ```

2. **Edit metadata.yaml** with your task information

3. **Choose subtasks to keep** - delete examples you don't need

4. **Customize each subtask**:
   - Update class names (PascalCase matching filename)
   - Set dataset path
   - Configure labels/schema/metrics
   - Customize prompts

5. **Test your task**:
   ```bash
   benchy eval my_task --limit 10
   ```

## What's Included

- `metadata.yaml` - Task group metadata and configuration
- `mcq_example.py` - Multiple choice classification template
- `structured_example.py` - Structured extraction template
- `freeform_example.py` - Freeform generation template

## Minimal Example

For a simple multiple choice task, you only need:

```python
# src/tasks/my_task/my_subtask.py
from ..formats import MultipleChoiceHandler

class MySubtask(MultipleChoiceHandler):
    dataset = "org/my-dataset"
    labels = {0: "No", 1: "Yes"}
    system_prompt = "You are a classifier."
```

That's it! **~5 lines** for a working task.

## Documentation

See [HANDLER_SYSTEM_GUIDE.md](../../../../docs/HANDLER_SYSTEM_GUIDE.md) for:
- Complete handler reference
- Customization examples
- Best practices
- Troubleshooting

## File Structure

```
src/tasks/my_task/
  ├── metadata.yaml          # Task group metadata
  ├── subtask_one.py        # Subtask 1 (handler class)
  ├── subtask_two.py        # Subtask 2 (handler class)
  └── .data/                # Cached datasets (auto-created)
```

## Handler Types

Choose the appropriate handler for your task format:

| Handler | Use Case | Example |
|---------|----------|---------|
| `MultipleChoiceHandler` | Classification, MCQ | Sentiment, NLI |
| `StructuredHandler` | Extraction, JSON output | Entity recognition |
| `FreeformHandler` | Generation, translation | Summarization |
| `MultimodalStructuredHandler` | Image → JSON | Invoice extraction |

## Running Tasks

```bash
# Run all subtasks
benchy eval my_task

# Run specific subtask
benchy eval my_task.subtask_one

# Test with limited samples
benchy eval my_task --limit 10
```

## Tips

1. **Start simple** - Use handler defaults first, customize only if needed
2. **Test incrementally** - Verify basic functionality before adding complexity
3. **Use meaningful names** - File names become task identifiers
4. **Document well** - Add clear docstrings, they show up in task listings
5. **Check metadata** - Ensure capability_requirements match your needs

## Migration from Old System

If converting from the old system:

1. Task group metadata → `metadata.yaml`
2. Each subtask → separate `.py` file with handler class
3. Custom metrics config → `metrics_config` attribute
4. Dataset loading → handler takes care of it automatically
5. Remove `task.json`, `run.py`, old `task.py`

See the [migration guide](../../../../docs/HANDLER_SYSTEM_GUIDE.md#migration-from-old-system) for details.

