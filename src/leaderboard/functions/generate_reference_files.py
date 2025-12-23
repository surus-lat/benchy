#!/usr/bin/env python3
"""
Generate reference task metadata files from per-task configs.
"""

import json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List


DEFAULT_GROUP_ORDER = [
    "latam_pr",
    "latam_es",
    "translation",
    "structured_extraction",
]


def _iter_task_configs(tasks_root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for config_path in sorted(tasks_root.rglob("task.json")):
        if config_path.parent.name == "_template":
            continue
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        yield config_path, config


def _build_reference_payloads(tasks_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tasks_list: Dict[str, Any] = {"tasks": {}}
    task_groups: Dict[str, Any] = {"task_groups": {}}

    grouped_configs: Dict[str, List[Dict[str, Any]]] = {}
    for _, config in _iter_task_configs(tasks_root):
        group_id = config.get("group")
        if not group_id:
            continue
        grouped_configs.setdefault(group_id, []).append(config)

    ordered_groups = [
        group_id
        for group_id in DEFAULT_GROUP_ORDER
        if group_id in grouped_configs
    ]
    ordered_groups.extend(
        group_id
        for group_id in sorted(grouped_configs)
        if group_id not in DEFAULT_GROUP_ORDER
    )

    for group_id in ordered_groups:
        for config in grouped_configs[group_id]:
            group_metadata = config.get("group_metadata", {})
            if group_metadata and group_id not in task_groups["task_groups"]:
                group_entry = dict(group_metadata)
                group_entry["subtasks"] = []
                task_groups["task_groups"][group_id] = group_entry

            tasks_in_group = config.get("tasks") or []
            task_metadata = config.get("task_metadata") or {}

            for task_name in tasks_in_group:
                metadata = task_metadata.get(task_name)
                if not metadata:
                    continue
                entry = {"name": task_name, "group": group_id}
                entry.update(metadata)

                if "URL" not in entry:
                    url = entry.pop("url", None) or entry.pop("Url", None)
                    if url:
                        entry["URL"] = url

                tasks_list["tasks"][task_name] = entry

                group_entry = task_groups["task_groups"].get(group_id)
                if group_entry is not None and task_name not in group_entry["subtasks"]:
                    group_entry["subtasks"].append(task_name)

    return tasks_list, task_groups


def generate_reference_files(reference_dir: str, tasks_root: str = None) -> bool:
    if tasks_root is None:
        tasks_root = Path(__file__).resolve().parents[2] / "tasks"
    else:
        tasks_root = Path(tasks_root)

    reference_path = Path(reference_dir)
    reference_path.mkdir(parents=True, exist_ok=True)

    tasks_list, task_groups = _build_reference_payloads(tasks_root)

    tasks_list_path = reference_path / "tasks_list.json"
    tasks_groups_path = reference_path / "tasks_groups.json"

    tasks_list_path.write_text(
        json.dumps(tasks_list, indent=2, ensure_ascii=True) + "\n"
    )
    tasks_groups_path.write_text(
        json.dumps(task_groups, indent=2, ensure_ascii=True) + "\n"
    )

    print(f"✓ Wrote {tasks_list_path}")
    print(f"✓ Wrote {tasks_groups_path}")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate reference task metadata files from task configs."
    )
    parser.add_argument(
        "--reference-dir",
        default="reference",
        help="Directory to write tasks_list.json and tasks_groups.json",
    )
    parser.add_argument(
        "--tasks-root",
        default=None,
        help="Root directory containing task.json configs (defaults to src/tasks)",
    )

    args = parser.parse_args()
    generate_reference_files(args.reference_dir, args.tasks_root)


if __name__ == "__main__":
    main()
