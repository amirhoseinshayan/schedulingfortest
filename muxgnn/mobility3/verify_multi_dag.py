"""Structural checks; optionally test a real SUMO trace without changing it.

The default temporary constant-position trace is ONLY a logic fixture, never
represented as measured mobility or used in scientific performance results.
"""
from __future__ import annotations
import argparse,csv,math,tempfile
from pathlib import Path

import networkx as nx
import numpy as np

from .multi_dag import DAGS_PER_EPISODE,EPISODE_DURATION_SEC,make_workload,episode_workloads
from .episode import SevenDagEpisode
from .simulator import ThreeEdgeMobilitySimulator,WorkflowTimeLimitReached
from .settings import SUMO_DIR
from .trace import MobilityTrace
from .scaled_source import load_scaled_task_graph


def verify_graphs():
    orig=load_scaled_task_graph()
    seen=set()
    for split in ('train','val','test'):
        for index in (0,1):
            gs=episode_workloads(split,index)
            assert len(gs)==DAGS_PER_EPISODE
            for g in gs:
                assert g['dag_fingerprint'] not in seen
                seen.add(g['dag_fingerprint'])
                assert len(g['profiles'])==60 and len(g['dependencies'])==126
                assert len(g['layers'])==8 and nx.is_directed_acyclic_graph(g['dag'])
                assert nx.is_weakly_connected(g['dag'])
                assert g['critical_path_sec']>0 and math.isclose(g['workflow_deadline_sec'],1.15*g['critical_path_sec'])
                assert g['dag_fingerprint']==make_workload(g['dag_seed'])['dag_fingerprint']
                for p,ref in zip(g['profiles'],orig['profiles']):
                    for field in ('a7_time_sec','a12_time_sec','a7_energy_j','a12_energy_j','gang_size','a7_power_w','a12_power_w'):
                        assert p[field]==ref[field],(field,g['dag_seed'])
                    assert math.isclose(p['deadline_sec'],1.15*p['dependency_earliest_finish_sec'],rel_tol=1e-9)
    print('PASS: 42 split-disjoint DAGs, original task profiles, deadlines, connectivity, seeds')


def fixture(path):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(('time_sec','vehicle_id','x_m','y_m','speed_mps','angle_deg'))
        for t in (0.,600.):
            for i in range(16):w.writerow((t,f'V{i}',500+i%4,700+i//4,0,0))


def verify_episode(path):
    trace=MobilityTrace(path,SUMO_DIR/'three_servers.json')
    ep=SevenDagEpisode(trace,'test',0,trace.times[0],4000)
    first_cap=(ep.sim.state.scenario_capacity_a7.copy(),ep.sim.state.scenario_capacity_a12.copy())
    times=[trace.times[0]]
    for i in range(7):
        sim=ep.sim
        for _ in sim.iter_decision_epochs():pass
        assert np.all(sim.state.completed) and not np.any(sim.awaiting_delivery)
        assert all(np.array_equal(a,b) for a,b in zip(first_cap,(sim.state.scenario_capacity_a7,sim.state.scenario_capacity_a12)))
        times.append(sim.state.physical_time_sec)
        if not ep.finish_dag():break
    assert len(ep.completed_dags)==7 and times==sorted(times)
    assert abs(ep.energy_episode_j-sum(x['energy_j'] for x in ep.completed_dags)) < 1e-6
    print('PASS: 7 sequential DAGs, 420 delivered tasks, time continuity, resource capacity, energy accounting')
    if ep.sim.state.physical_time_sec>ep.end_sec+1e-8:
        raise AssertionError('Overran SUMO time limit')
    # Separate short horizon test: never pretend a partly executed task was delivered.
    last=make_workload(555777)
    cutoff=ThreeEdgeMobilitySimulator(trace,42,42,'earliest_finish',trace.end-1.,workload=last,episode_end_sec=trace.end)
    try:
        for _ in cutoff.iter_decision_epochs():pass
    except WorkflowTimeLimitReached:
        assert abs(cutoff.state.physical_time_sec-trace.end)<1e-7
        assert not np.all(cutoff.state.completed)
        assert all(x>=0 for x in cutoff.energy_spent_j)
        print('PASS: 600s truncation retains partial energy and does not fake completion')
    else:
        raise AssertionError('Time-limit fixture unexpectedly finished all 60 tasks in 1s')


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,default=None)
    args=p.parse_args()
    verify_graphs()
    if args.trace is not None:
        verify_episode(args.trace)
        print('Note: real mobility trace supplied; successful run still needs PPO training evaluation')
    else:
        with tempfile.TemporaryDirectory() as work:
            t=Path(work)/'synthetic_fixture.csv';fixture(t);verify_episode(t)
        print('Note: only a constant-position logic fixture was used; this is not SUMO validation')

if __name__=='__main__':main()
