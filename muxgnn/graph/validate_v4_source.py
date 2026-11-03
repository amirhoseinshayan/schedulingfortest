from __future__ import annotations

import math

import networkx as nx

from .v4_settings import V4_DEADLINE_SLACK_FACTOR, V4_RELEASE_TIME_SEC
from .v4_source import load_v4_physical_task_graph


def main() -> None:
    source = load_v4_physical_task_graph()
    dag = source["dag"]
    profiles = list(source["profiles"])

    assert len(profiles) == 60
    assert len(source["dependencies"]) == 126
    assert nx.is_directed_acyclic_graph(dag)
    assert nx.is_weakly_connected(dag)
    assert len(source["layers"]) == 8
    assert all(abs(float(p["release_time_sec"]) - V4_RELEASE_TIME_SEC) < 1e-15 for p in profiles)
    assert all(abs(float(p["period_placeholder"]) - 1.0) < 1e-15 for p in profiles)

    for p in profiles:
        assert math.isclose(float(p["a7_time_sec"]), float(p["a7_execution_time_ms"]) / 1000.0, rel_tol=0, abs_tol=1e-15)
        assert math.isclose(float(p["a12_time_sec"]), float(p["a12_execution_time_ms"]) / 1000.0, rel_tol=0, abs_tol=1e-15)
        # m_i is occupancy only: the compatibility gang-time aliases must equal
        # the measured task durations exactly, never T/m_i.
        assert math.isclose(float(p["a7_gang_time_sec"]), float(p["a7_time_sec"]), rel_tol=0, abs_tol=1e-15)
        assert math.isclose(float(p["a12_gang_time_sec"]), float(p["a12_time_sec"]), rel_tol=0, abs_tol=1e-15)
        assert math.isclose(float(p["a7_energy_j"]), float(p["a7_time_sec"]) * float(p["a7_power_w"]), rel_tol=1e-12)
        assert math.isclose(float(p["a12_energy_j"]), float(p["a12_time_sec"]) * float(p["a12_power_w"]), rel_tol=1e-12)
        assert float(p["deadline_sec"]) >= float(p["dependency_earliest_finish_sec"])
        assert math.isclose(
            float(p["deadline_sec"]),
            V4_DEADLINE_SLACK_FACTOR * float(p["dependency_earliest_finish_sec"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )

    print("v4 physical source validation: PASS")
    print(f"Tasks                  : {len(profiles)}")
    print(f"Dependencies           : {len(source['dependencies'])}")
    print(f"Topological layers     : {len(source['layers'])}")
    print(f"Critical path (s)      : {source['critical_path_sec']:.9f}")
    print(f"Workflow deadline (s)  : {source['workflow_deadline_sec']:.9f}")
    print("External release       : 0 s for all tasks")
    print("Raw execution unit     : ms -> stored scheduling unit: s")
    print("Period=1 semantics     : placeholder only; not arrival/release period")
    print("Deadline dependence    : DAG timing only; no HEFT/PPO schedule")
    print("Gang runtime semantics : Runtime=T; m_i only reserves simultaneous cores")


if __name__ == "__main__":
    main()
