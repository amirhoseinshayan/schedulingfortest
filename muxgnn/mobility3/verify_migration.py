"""Deterministic simulator-level contract checks without SUMO or PPO runtime.

Run: python -m muxgnn.mobility3.verify_migration
Tests cover zero transfer, saved progress, stale completions, buffered results,
no migration across disconnected source, exact capacity/energy conservation.
"""
from __future__ import annotations
import math
import numpy as np
from .simulator import ThreeEdgeMobilitySimulator

class SyntheticTrace:
    times=[0.,500.]
    end=500.
    servers=[{'x':0.,'y':0.,'range_m':300.}]*4
    def __init__(self,events=(),offline=None):
        self.events=list(events)
        self.offline=offline or (lambda t:set())
    def at(self,t):
        online=set(range(1,16))-set(self.offline(t))
        return {'v2v_candidates':online,'edge_candidates':{0,1,2,3},
                'vehicles':{f'V{i}':{'x':0.,'y':0.} for i in range(16)}}

def start_on_v1(sim):
    action=next(a for a in sim._valid_actions() if a.task_id==6 and a.resource_slot==1 and a.processor_index==0)
    sim._start_task(action)
    return action

def test_instant_migration():
    tr=SyntheticTrace(events=[(0.5,[{'event':'LINK_DOWN','link_type':'V2V','source':'V0','target':'V15'}])])
    sim=ThreeEdgeMobilitySimulator(tr,start_sec=0.)
    original=start_on_v1(sim)
    free_before=int(sim.state.free_a7[1])
    event=sim.advance_one_event()
    assert sim.state.simulation_time_sec == 0.5
    choices=sim.valid_migrations()
    choice=next(c for c in choices if c.task_id==6 and c.resource_slot==2 and c.processor_index==1)
    assert 0<choice.progress<1
    m=sim.apply_migration(choice)
    assert m['transfer_time_sec']==0. and m['transfer_energy_j']==0.
    assert int(sim.migration_count[6])==1
    assert sim.state.free_a7[1]==free_before+original.cores
    assert sim.state.assigned_resource_slot[6]==2
    assert math.isclose(sim.work_done[6],0.5/original.duration_sec,rel_tol=1e-9)
    expected_remaining=(1.-sim.work_done[6])*sim._task_duration(6,1)
    assert math.isclose(sim.state.task_finish_time_sec[6],0.5+expected_remaining,rel_tol=1e-9)
    while not sim.state.completed[6]:
        sim.advance_one_event()
    expected_energy=0.5*sim._task_power(6,0)+expected_remaining*sim._task_power(6,1)
    assert math.isclose(sim.energy_spent_j[6],expected_energy,rel_tol=1e-8)
    assert math.isclose(sim.delivery_time_sec[6],0.5+expected_remaining,rel_tol=1e-8)
    assert sim.state.free_a12[2]<=sim.state.scenario_capacity_a12[2]
    assert sim.state.free_a7[1]==free_before+original.cores


def test_disconnect_buffers_result():
    tr=SyntheticTrace(events=[
        (1.0,[{'event':'LINK_DOWN','link_type':'V2V','source':'V0','target':'V1'}]),
        (3.0,[{'event':'LINK_UP','link_type':'V2V','source':'V0','target':'V1'}])],
        offline=lambda t:{1} if 1.0<=t<3.0 else set())
    sim=ThreeEdgeMobilitySimulator(tr,start_sec=0.)
    start_on_v1(sim)
    a=sim.advance_one_event()
    assert sim.state.simulation_time_sec==1.
    assert all(c.is_keep for c in sim.valid_migrations() if c.task_id==6)
    # If the radio is lost, running computation continues uninterrupted.
    sim.apply_migration(next(c for c in sim.valid_migrations() if c.task_id==6 and c.is_keep))
    while sim.state.simulation_time_sec<2.245-1e-9:
        sim.advance_one_event()
    assert sim.awaiting_delivery[6] and not sim.state.completed[6]
    assert not sim.state.running[6]
    assert math.isclose(sim.state.task_finish_time_sec[6],2.245,rel_tol=1e-9)
    # Predecessor task completion cannot release successors until data arrives.
    assert not any(sim.state.ready[i] for i in sim.successors[6])
    while not sim.state.completed[6]:sim.advance_one_event()
    assert math.isclose(sim.delivery_time_sec[6],3.,rel_tol=1e-9)
    assert sim.state.completed[6]
    assert all(sim.state.ready[i] for i in sim.successors[6] if all(sim.state.completed[p] for p in sim.predecessors[i]))


def test_run_full_episode():
    tr=SyntheticTrace(events=[
        (2.,[{'event':'LINK_DOWN','link_type':'V2V','source':'V0','target':'V1'}]),
        (6.,[{'event':'LINK_UP','link_type':'V2V','source':'V0','target':'V1'}])],
        offline=lambda t:{1} if 2<=t<6 else set())
    sim=ThreeEdgeMobilitySimulator(tr,start_sec=.1)
    epochs=0
    for _ in sim.iter_decision_epochs():
        epochs+=1
        assert epochs<10000
        assert np.all(sim.state.free_a7>=0) and np.all(sim.state.free_a12>=0)
        assert np.all(sim.state.free_a7<=sim.state.scenario_capacity_a7)
        assert np.all(sim.state.free_a12<=sim.state.scenario_capacity_a12)
    assert int(sum(sim.state.completed))==60
    assert len(sim.delivery_history)==60
    assert np.all(sim.delivery_time_sec>=sim.state.task_start_time_sec)
    assert np.all(sim.energy_spent_j>0)
    assert epochs>60
    return epochs

if __name__=='__main__':
    test_instant_migration()
    print('PASS: instant migration / work and energy conservation / stale events')
    test_disconnect_buffers_result()
    print('PASS: link loss / buffered output / dependency causality')
    n=test_run_full_episode()
    print(f'PASS: 60-task full scheduling baseline, {n} decision epochs, core bounds')
