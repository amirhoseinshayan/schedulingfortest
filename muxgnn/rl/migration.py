from __future__ import annotations

from typing import Tuple

from .config import MUX_MIGRATION_FEATURE_DIM


# Runtime-only features used by the migration Transformer. They do not alter
# the stored heterogeneous graph or the offline muxGNN pretraining contract.
MIGRATION_FEATURE_NAMES: Tuple[str, ...] = (
    "progress",
    "remaining_fraction",
    "migration_count_ratio",
    "state_size_norm",
    "source_resource_norm",
    "source_is_a12",
    "target_resource_norm",
    "target_is_a12",
    "is_keep",
    "target_available_ratio",
    "target_queue_delay_norm",
    "target_compute_scale",
    "target_power_scale",
    "reachable",
    "link_rate_norm",
    "transfer_delay_norm",
    "contact_duration_norm",
    "remaining_exec_time_norm",
    "estimated_finish_deadline_ratio",
    "migration_energy_norm",
)

if len(MIGRATION_FEATURE_NAMES) != MUX_MIGRATION_FEATURE_DIM:
    raise RuntimeError(
        "MUX_MIGRATION_FEATURE_DIM does not match MIGRATION_FEATURE_NAMES."
    )
