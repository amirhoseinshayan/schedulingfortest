"""Continuous 600 s trace, sequential 7-DAG episodes, fixed resource capacities.

A new per-DAG *relative task clock* starts at zero, but its SUMO absolute time
starts at the preceding DAG's completion timestamp. No vehicle/trace restart.
The PPO policy and optimizer live above this environment and are never reset.
"""
from __future__ import annotations

import numpy as np

from .multi_dag import DAGS_PER_EPISODE, EPISODE_DURATION_SEC, episode_workloads, dag_summary
from .simulator import ThreeEdgeMobilitySimulator


class SevenDagEpisode:
    def __init__(self, trace, split, episode_index, start_sec, seed, behavior_policy='earliest_finish', simulator_class=ThreeEdgeMobilitySimulator):
        self.trace=trace
        self.split=split
        self.episode_index=int(episode_index)
        self.start_sec=float(start_sec)
        self.end_sec=self.start_sec+EPISODE_DURATION_SEC
        if self.start_sec < trace.times[0]-1e-8 or self.end_sec > trace.end+1e-8:
            raise ValueError('One 600 s uninterrupted SUMO trace is required per episode')
        self.seed=int(seed)
        self.policy=str(behavior_policy)
        self.simulator_class=simulator_class
        self.workloads=episode_workloads(split,self.episode_index)
        self.completed_dags=[]
        self.incomplete_dags=[]
        self.energy_episode_j=0.
        self.reward_episode=0.
        self.dag_index=0
        self.sim=self._make_sim(None,self.start_sec)

    def _make_sim(self, inherited, release_sec):
        capacities=None if inherited is None else {
            'a7': inherited.state.scenario_capacity_a7.copy(),
            'a12': inherited.state.scenario_capacity_a12.copy(),
        }
        sim=self.simulator_class(self.trace,scenario_id=self.episode_index*DAGS_PER_EPISODE+self.dag_index,
            seed=self.seed,behavior_policy=self.policy,start_sec=release_sec,
            workload=self.workloads[self.dag_index],capacity_override=capacities,
            episode_end_sec=self.end_sec)
        if inherited is not None:
            if np.any(inherited.state.running) or np.any(inherited.awaiting_delivery):
                raise AssertionError('Cannot release DAG while previous work/results are pending')
            for name in ('a7','a12'):
                if not np.array_equal(getattr(inherited.state,'free_'+name),getattr(inherited.state,'scenario_capacity_'+name)):
                    raise AssertionError('Task cores were not fully returned at DAG completion')
                if not np.array_equal(getattr(sim.state,'scenario_capacity_'+name),getattr(inherited.state,'scenario_capacity_'+name)):
                    raise AssertionError('Capacity changed at DAG transition')
            if abs(sim.state.physical_time_sec - inherited.state.physical_time_sec)>1e-8:
                raise AssertionError('Global SUMO time was restarted or skipped')
        return sim

    def finish_dag(self, reward_total=0.):
        """Call only after ALL 60 task results have been delivered."""
        sim=self.sim
        if not bool(np.all(sim.state.completed)) or np.any(sim.awaiting_delivery):
            raise AssertionError('Cannot mark an unfinished DAG complete')
        energy=float(sum(sim.energy_spent_j))
        self.energy_episode_j+=energy
        self.reward_episode+=float(reward_total)
        self.completed_dags.append({**dag_summary(self.workloads[self.dag_index]),
            'index':self.dag_index+1,'start_sumo_sec':sim.start_sec,
            'end_sumo_sec':float(sim.state.physical_time_sec),
            'makespan_sec':float(sim.state.simulation_time_sec),
            'energy_j':energy, 'reward_sum':float(reward_total), 'num_completed_tasks':int(sum(sim.state.completed)),
            'migrations':int(sum(sim.migration_count)), 'success':True})
        if self.dag_index+1==DAGS_PER_EPISODE:
            return False
        if sim.state.physical_time_sec >= self.end_sec-1e-9:
            next_g=self.workloads[self.dag_index+1]
            self.incomplete_dags.append({**dag_summary(next_g),'index':self.dag_index+2,
                'start_sumo_sec':None,'end_sumo_sec':self.end_sec,
                'num_completed_tasks':0,'num_running_tasks':0,
                'num_buffered_outputs':0,'energy_j':0.,
                'reason':'next_dag_not_released_episode_time_exhausted','success':False})
            return False
        self.dag_index+=1
        self.sim=self._make_sim(sim,float(sim.state.physical_time_sec))
        return True

    def record_incomplete(self, reason='time_limit'):
        sim=self.sim
        self.incomplete_dags.append({**dag_summary(self.workloads[self.dag_index]),
            'index':self.dag_index+1, 'start_sumo_sec':sim.start_sec,
            'end_sumo_sec':float(sim.state.physical_time_sec),
            'num_completed_tasks':int(sum(sim.state.completed)),
            'num_running_tasks':int(sum(sim.state.running)),
            'num_buffered_outputs':int(sum(sim.awaiting_delivery)),
            'energy_j':float(sum(sim.energy_spent_j)), 'reason':reason, 'success':False})
        return self.report()

    def report(self):
        current_energy=float(sum(self.sim.energy_spent_j)) if (self.incomplete_dags and self.incomplete_dags[-1].get('start_sumo_sec') is not None) else 0.
        return {'episode_index':self.episode_index,'split':self.split,
                'episode_start_sumo_sec':self.start_sec,'episode_end_limit_sec':self.end_sec,
                'completed_dag_count':len(self.completed_dags),
                'planned_dag_count':DAGS_PER_EPISODE,'completed_dags':self.completed_dags,
                'incomplete_dags':self.incomplete_dags,
                'energy_episode_j':self.energy_episode_j+current_energy,
                'all_dags_completed':len(self.completed_dags)==DAGS_PER_EPISODE}
