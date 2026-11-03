"""A separate, derived 5x physical-duration workload; original v4 remains untouched.

All A7/A12 execution times, DAG-based deadlines, and energy in joules are
scaled by the same factor. Average power, gang width, task precedence and
external release are unchanged. This is a *synthetic longer-workload scenario*,
NOT a claim that A7/A12 measurements were performed again.
"""
from __future__ import annotations
from functools import lru_cache
from ..graph.v4_source import load_v4_physical_task_graph
from .settings import TIME_SCALE_FACTOR

# Use ONLY fields with physical time dimensions; don't change dimensionless DAG/gang data.
_TIME_FIELDS = (
    'a7_time_sec', 'a12_time_sec', 'a7_gang_time_sec', 'a12_gang_time_sec',
    'fastest_gang_time_sec', 'dependency_earliest_ready_sec',
    'dependency_earliest_finish_sec', 'deadline_sec',
    'deadline_slack_from_dependency_earliest_finish_sec',
)

@lru_cache(maxsize=1)
def load_scaled_task_graph():
    ref = load_v4_physical_task_graph()
    k = float(TIME_SCALE_FACTOR)
    if not k > 0: raise ValueError('TIME_SCALE_FACTOR must be positive')
    source = dict(ref)
    profiles = []
    for p in ref['profiles']:
        q = dict(p)
        q['reference_v4_a7_time_sec'] = float(p['a7_time_sec'])
        q['reference_v4_a12_time_sec'] = float(p['a12_time_sec'])
        q['reference_v4_deadline_sec'] = float(p['deadline_sec'])
        q['reference_v4_a7_energy_j'] = float(p['a7_energy_j'])
        q['reference_v4_a12_energy_j'] = float(p['a12_energy_j'])
        q['reference_v4_a7_execution_time_ms'] = float(p['a7_execution_time_ms'])
        q['reference_v4_a12_execution_time_ms'] = float(p['a12_execution_time_ms'])
        for field in _TIME_FIELDS:
            q[field] = float(p[field]) * k
        # These *_ms fields denote effective scenario times in snapshots;
        # raw benchmark data live only in reference_v4_* and the original files.
        q['a7_execution_time_ms'] = float(p['a7_execution_time_ms']) * k
        q['a12_execution_time_ms'] = float(p['a12_execution_time_ms']) * k
        # Consistency E = T x P; power remains measured-reference assumption.
        q['a7_energy_j'] = q['a7_time_sec'] * float(q['a7_power_w'])
        q['a12_energy_j'] = q['a12_time_sec'] * float(q['a12_power_w'])
        profiles.append(q)
    source['profiles'] = profiles
    source['critical_path_sec'] = float(ref['critical_path_sec']) * k
    source['workflow_deadline_sec'] = float(ref['workflow_deadline_sec']) * k
    source['time_scale_factor'] = k
    source['reference_v4_critical_path_sec'] = float(ref['critical_path_sec'])
    source['reference_v4_workflow_deadline_sec'] = float(ref['workflow_deadline_sec'])
    return source
