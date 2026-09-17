"""Consistency tests for Facturas Argentinas dataset identifiers."""

import ast

import yaml

def test_facturas_dataset_repo_id_matches_metadata():
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

    url = metadata["subtasks"]["facturas_argentinas"]["dataset_url"]
    repo_from_url = url.rsplit("/datasets/", 1)[-1].strip("/")

    assert dataset_repo_id == repo_from_url
    assert dataset_repo_id == "mauroibz/facturas_argentinas_2"
