"""Legacy-native Simple MAGNET heterogeneous graph dataset utilities."""

from .builder import build_hetero_graph, load_legacy_native_task_graph

__all__ = [
    "build_hetero_graph",
    "load_legacy_native_task_graph",
]
