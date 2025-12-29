# Task Migration Guide

This guide explains how to migrate older task wrappers to the new TaskGroupRunner
format. The goal is to remove boilerplate and keep task-specific logic isolated
to task factories and aggregation functions.

## Old Pattern (Legacy)

Typical legacy `run_<task>()` wrappers included:
- provider type resolution
- connection info creation
- output directory setup
- subtask loop
- interface construction
- BenchmarkRunner + save_results
- aggregation + summary

## New Pattern (TaskGroupRunner)

You now define:
1. A `TaskGroupSpec` describing the task.
2. A `prepare_task()` factory (and optional `run_subtask()` for custom loops).
3. An `aggregate_metrics()` function.
4. A `write_summary()` function (usually `write_group_summary`).

Then the `run_<task>()` wrapper simply calls `run_task_group(...)`.

## Migration Steps

1. **Create a TaskGroupSpec**
   - Set `name`, `display_name`, `output_subdir`.
   - Provide `prepare_task` or `run_subtask`.
   - Attach `aggregate_metrics` and `write_summary`.
   - Add `setup` / `teardown` if you need shared resources across subtasks.

2. **Move task construction into `prepare_task()`**
   - Use `SubtaskContext` to access `task_config`, `prompts`, and `subtask_config`.

3. **Move aggregation into `aggregate_metrics()`**
   - Reuse your existing logic unchanged.

4. **Update the wrapper**
   - Replace the old orchestration with `return run_task_group(...)`.
   - If you preload shared resources, wire them into `setup` and pass through `context.shared`.

5. **Register the task**
   - Add the task to `src/pipeline.py` `TASK_REGISTRY`.

## Example

```python
def _prepare_my_task(context: SubtaskContext):
    return MyTask({
        "dataset": context.subtask_config,
        "prompts": context.prompts,
    })

MY_TASK_SPEC = TaskGroupSpec(
    name="my_task",
    display_name="My Task",
    output_subdir="my_task",
    prepare_task=_prepare_my_task,
    aggregate_metrics=_aggregate_my_metrics,
    write_summary=_write_summary,
)

@task
def run_my_task(...):
    return run_task_group(
        spec=MY_TASK_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )
```

## References

- `src/tasks/TASK_TEMPLATE.md`
- `src/tasks/spanish/run.py`
- `src/tasks/structured/run.py`
