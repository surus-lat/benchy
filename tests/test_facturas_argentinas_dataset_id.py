"""Consistency tests for Facturas Argentinas dataset identifiers."""

import ast

import yaml

def test_facturas_dataset_repo_id_matches_metadata():
    """
    Verify that the dataset repository identifier declared in the FacturasArgentinas source matches the repository derived from the metadata and the expected repository id.
    
    This test extracts the `dataset_repo_id` declared on the `FacturasArgentinas` class, reads the `facturas_argentinas` subtask metadata (accepting `URL`, `url`, `Url`, or `dataset_url`), derives the repository identifier from the metadata URL, and asserts that the two identifiers match and equal "mauroibz/facturas_argentinas_2".
    """
    with open("src/tasks/document_extraction/facturas_argentinas.py", "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    dataset_repo_id = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FacturasArgentinas":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "dataset_repo_id":
                            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                                dataset_repo_id = stmt.value.value
                                break
                if dataset_repo_id is not None:
                    break
        if dataset_repo_id is not None:
            break

    assert dataset_repo_id is not None

    with open("src/tasks/document_extraction/metadata.yaml", "r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    task_metadata = metadata["subtasks"]["facturas_argentinas"]
    url = (
        task_metadata.get("URL")
        or task_metadata.get("url")
        or task_metadata.get("Url")
        or task_metadata.get("dataset_url")
    )
    assert url is not None
    repo_from_url = url.rsplit("/datasets/", 1)[-1].strip("/")

    assert dataset_repo_id == repo_from_url
    assert dataset_repo_id == "mauroibz/facturas_argentinas_2"
