"""Generic benchmark execution engine.

This module provides the core infrastructure for running benchmarks:
- BenchmarkRunner: Generic runner that works with any task + interface
- Protocols: BaseTask and BaseInterface contracts
- Connection utilities: Build connection_info from provider configs
- Checkpoint utilities: For resumable benchmark runs
- Result handling and aggregation
"""

from .benchmark_runner import BenchmarkRunner, run_benchmark, save_results
from .protocols import BaseTask, BaseInterface, check_compatibility
from .connection import build_connection_info, get_interface_for_provider
from .checkpoint import (
    get_checkpoint_path,
    get_config_hash,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Runner
    "BenchmarkRunner",
    "run_benchmark",
    "save_results",
    # Protocols
    "BaseTask",
    "BaseInterface",
    "check_compatibility",
    # Connection
    "build_connection_info",
    "get_interface_for_provider",
    # Checkpoint
    "get_checkpoint_path",
    "get_config_hash",
    "save_checkpoint",
    "load_checkpoint",
]
