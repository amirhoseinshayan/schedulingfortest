from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, expected: str) -> str | None:
    target = expected.strip().lower()
    for column in df.columns:
        if str(column).strip().lower() == target:
            return str(column)
    return None


def find_processor_folder(results_root: Path, workload_group: str, processor: str) -> Path:
    group_root = results_root / workload_group
    if not group_root.exists():
        raise FileNotFoundError(f"Workload group not found: {group_root}")
    candidates = [
        path
        for path in group_root.rglob(processor)
        if path.is_dir() and any(path.glob("*.csv"))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {processor} benchmark folder under {group_root}, "
            f"found: {candidates}"
        )
    return candidates[0]


def _validate_sample_index(df: pd.DataFrame, file_path: Path) -> int:
    """Validate that '#' is only a contiguous sample counter and return N."""
    index_col = _find_column(df, "#")
    if index_col is None:
        raise ValueError(f"Missing '#' sample-index column: {file_path}")

    values = pd.to_numeric(df[index_col], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Non-numeric sample index in {file_path}")
    observed = values.to_numpy(dtype=float)
    expected = np.arange(1, len(df) + 1, dtype=float)
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"Sample index in {file_path} is not the contiguous sequence 1..N; "
            "the trace cannot be converted to a duration safely."
        )
    return int(len(df))


def extract_processor_data(folder_path: Path) -> Dict[str, dict]:
    """Read raw power traces without assigning any unsupported wall-clock unit.

    The raw archive contains sample indices but no timestamps or acquisition
    period.  This function therefore records sample_count and measured power
    only.  A single common time calibration is applied later, after A7/A12
    profiles have been paired.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {folder_path}")

    task_data: Dict[str, dict] = {}
    for file_path in sorted(folder_path.glob("*.csv"), key=lambda p: p.name.lower()):
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"Empty benchmark CSV: {file_path}")

        sample_count = _validate_sample_index(df, file_path)

        dynamic_col = _find_column(df, "Dynamic")
        static_col = _find_column(df, "Static")
        total_col = _find_column(df, "Total")

        avg_dynamic = 0.0
        avg_static = 0.0
        avg_total = None

        if dynamic_col is not None:
            dynamic_values = pd.to_numeric(df[dynamic_col], errors="coerce").dropna()
            if not dynamic_values.empty:
                avg_dynamic = float(dynamic_values.mean())

        if static_col is not None:
            static_values = pd.to_numeric(df[static_col], errors="coerce").dropna()
            if not static_values.empty:
                avg_static = float(static_values.mean())

        if total_col is not None:
            total_values = pd.to_numeric(df[total_col], errors="coerce").dropna()
            if not total_values.empty:
                avg_total = float(total_values.mean())

        # Preserve the project's existing power semantics unchanged.
        if dynamic_col is not None and static_col is not None:
            avg_total_power = avg_dynamic + avg_static
            if avg_total is not None and not np.isclose(
                avg_total_power, avg_total, rtol=1e-5, atol=1e-8
            ):
                raise ValueError(
                    f"Static+Dynamic does not match Total mean in {file_path}: "
                    f"{avg_total_power} vs {avg_total}"
                )
        elif avg_total is not None:
            avg_total_power = avg_total
        else:
            raise ValueError(
                f"{file_path} must contain Dynamic+Static or Total power columns."
            )

        if sample_count <= 0 or avg_total_power <= 0.0:
            raise ValueError(f"Non-positive sample count/power in benchmark: {file_path}")

        task_data[file_path.name.lower()] = {
            "file_name": file_path.name,
            "sample_count": int(sample_count),
            "avg_total_power_max_vf": float(avg_total_power),
            "avg_dynamic_power_max_vf": float(avg_dynamic),
            "avg_static_power_max_vf": float(avg_static),
        }

    if not task_data:
        raise ValueError(f"No CSV files found in {folder_path}")
    return task_data


def pair_processor_data(a7_data: Dict[str, dict], a12_data: Dict[str, dict]) -> List[dict]:
    common_names = sorted(set(a7_data).intersection(a12_data))
    if not common_names:
        raise ValueError("No matching A7/A12 benchmark filenames were found.")

    missing_on_a12 = sorted(set(a7_data) - set(a12_data))
    missing_on_a7 = sorted(set(a12_data) - set(a7_data))
    if missing_on_a12 or missing_on_a7:
        raise ValueError(
            "A7/A12 benchmark sets are not identical. "
            f"Missing on A12={missing_on_a12}; missing on A7={missing_on_a7}"
        )

    return [
        {"benchmark_name": name, "A7": dict(a7_data[name]), "A12": dict(a12_data[name])}
        for name in common_names
    ]


def calibrate_execution_times(
    paired_data: List[dict],
    physical_sample_interval_sec: float | None = None,
) -> Tuple[List[dict], dict]:
    """Assign execution times with no hidden sampling-period assumption.

    If a verified physical acquisition interval is explicitly supplied, true
    wall-clock seconds are used.  Otherwise the raw archive cannot support an
    absolute seconds claim, so a deterministic canonical time base is used:

        median A12 max-VF sample count == 1 canonical simulation second

    Every A7/A12 trace is multiplied by the same factor, preserving all timing
    ratios and scheduling order while removing dependence on an unknown
    acquisition interval.  The legacy CSV schema is left unchanged.
    """
    if not paired_data:
        raise ValueError("paired_data must not be empty")

    if physical_sample_interval_sec is not None:
        if physical_sample_interval_sec <= 0.0:
            raise ValueError("physical_sample_interval_sec must be positive")
        seconds_per_sample = float(physical_sample_interval_sec)
        calibration = {
            "mode": "physical_wall_clock_seconds",
            "physical_seconds_claimed": True,
            "seconds_per_sample": seconds_per_sample,
            "source": "explicit user-supplied --sample-interval-sec",
            "formula": "execution_time_sec = sample_count * seconds_per_sample",
        }
    else:
        a12_counts = np.asarray(
            [float(pair["A12"]["sample_count"]) for pair in paired_data], dtype=float
        )
        reference_sample_count = float(np.median(a12_counts))
        if not np.isfinite(reference_sample_count) or reference_sample_count <= 0.0:
            raise ValueError("Could not compute a valid canonical timing reference")
        seconds_per_sample = 1.0 / reference_sample_count
        calibration = {
            "mode": "canonical_relative_time",
            "physical_seconds_claimed": False,
            "reference_processor": "A12",
            "reference_statistic": "median max-VF raw sample count across matched profiles",
            "reference_sample_count": reference_sample_count,
            "canonical_reference_execution_time_sec": 1.0,
            "canonical_seconds_per_sample": seconds_per_sample,
            "formula": (
                "execution_time_canonical_sec = sample_count / "
                "median_A12_sample_count"
            ),
            "source": "derived only from raw trace sample counts; no acquisition-period assumption",
            "note": (
                "Values stored in legacy '*_sec' fields are canonical simulation seconds, "
                "not measured wall-clock seconds. A single common scale preserves all A7/A12 "
                "runtime ratios. Supply --sample-interval-sec only when the real acquisition "
                "period is known and verified."
            ),
        }

    calibrated: List[dict] = []
    for pair in paired_data:
        new_pair = {"benchmark_name": pair["benchmark_name"]}
        for processor in ("A7", "A12"):
            profile = dict(pair[processor])
            execution_time = float(profile["sample_count"]) * seconds_per_sample
            if not np.isfinite(execution_time) or execution_time <= 0.0:
                raise ValueError(
                    f"Invalid calibrated execution time for {pair['benchmark_name']} {processor}"
                )
            profile["time_scale_sec_per_sample"] = float(seconds_per_sample)
            profile["time_basis"] = calibration["mode"]
            profile["execution_time_max_vf"] = float(execution_time)
            new_pair[processor] = profile
        calibrated.append(new_pair)

    return calibrated, calibration


def profile_report_rows(paired_data: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for pair in paired_data:
        a7, a12 = pair["A7"], pair["A12"]
        rows.append(
            {
                "benchmark_name": pair["benchmark_name"],
                "time_basis": a7["time_basis"],
                "time_scale_sec_per_sample": a7["time_scale_sec_per_sample"],
                "a7_sample_count": a7["sample_count"],
                "a7_execution_time_sec": a7["execution_time_max_vf"],
                "a7_avg_total_power": a7["avg_total_power_max_vf"],
                "a12_sample_count": a12["sample_count"],
                "a12_execution_time_sec": a12["execution_time_max_vf"],
                "a12_avg_total_power": a12["avg_total_power_max_vf"],
                "a7_to_a12_time_ratio": (
                    a7["execution_time_max_vf"] / a12["execution_time_max_vf"]
                ),
            }
        )
    return rows
