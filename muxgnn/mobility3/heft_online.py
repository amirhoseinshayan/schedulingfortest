"""Same 7 DAGs, same 600s exogenous SUMO trace for online HEFT/PPO comparison.

This is the inherited causal, upward-rank, online HEFT-style baseline, not a
future-omniscient full offline classic HEFT. DAG releases occur immediately
on completion of previous DAG, hence *may differ in absolute time by policy*.
All released DAGs and trace timestamps are exogenously equal between methods.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import math
from ..rl.config import MUX_QOS_FACTOR
from .trace import MobilityTrace
from .simulator import ThreeEdgeMobilitySimulator,WorkflowTimeLimitReached
from .episode import SevenDagEpisode
from .settings import SUMO_DIR,DATASET_DIR,NUM_VEHICLES,NUM_EDGE_SERVERS

class OnlineHEFTThreeEdge(ThreeEdgeMobilitySimulator):
    def __init__(self,trace,scenario_id=0,seed=4000,start_sec=0,**kwargs):
        super().__init__(trace,scenario_id,seed,'earliest_finish',start_sec,**kwargs)
        weights=[(float(p['a7_time_sec'])+float(p['a12_time_sec']))/2 for p in self.profiles]
        ranks=[None]*self.num_tasks
        def rank(i):
            if ranks[i] is None:ranks[i]=weights[i]+max([rank(x) for x in self.successors[i]] or [0.0])
            return ranks[i]
        self.rank=[rank(i) for i in range(self.num_tasks)]
    def _select_action(self):
        actions=self._valid_actions()
        if not actions:return None
        now=float(self.state.simulation_time_sec)
        return min(actions,key=lambda a:(-self.rank[a.task_id],now+a.duration_sec,
                      a.energy_j,a.task_id,a.resource_slot,a.processor_index))

def run(trace, behavior, seed, scenario_id, start, split='test'):
    sim_cls=OnlineHEFTThreeEdge if behavior=='heft_online' else ThreeEdgeMobilitySimulator
    policy='earliest_finish' if behavior=='heft_online' else behavior
    episode=SevenDagEpisode(trace,split,scenario_id,start,seed,policy,simulator_class=sim_cls)
    counts={f'V{i}':0 for i in range(NUM_VEHICLES)}
    counts.update({f'EDGE{i}':0 for i in range(NUM_EDGE_SERVERS)})
    total_actions=0
    for dag_index in range(7):
        sim=episode.sim
        incomplete_reason=None
        try:
            for e in sim.iter_decision_epochs():
                if e.started_task is not None:
                    slot=int(sim.state.assigned_resource_slot[e.started_task])
                    counts[f'V{slot}' if slot<NUM_VEHICLES else f'EDGE{slot-NUM_VEHICLES}']+=1
                    total_actions+=1
        except WorkflowTimeLimitReached:
            incomplete_reason='600_second_time_limit'
        except RuntimeError as exc:
            incomplete_reason=str(exc)
        if incomplete_reason or not bool(np.all(sim.state.completed)):
            episode.record_incomplete(incomplete_reason or 'unfinished')
            break
        max_e=max(max(p['a7_energy_j'],p['a12_energy_j']) for p in sim.profiles)
        reward_sum=0.
        for task in range(sim.num_tasks):
            finish=float(sim.delivery_time_sec[task]);deadline=float(sim.deadline_sec[task])
            if finish<=deadline:qos=1.
            elif finish>=MUX_QOS_FACTOR*deadline:qos=0.
            else:qos=(MUX_QOS_FACTOR*deadline-finish)/max((MUX_QOS_FACTOR-1)*deadline,1e-9)
            reward_sum+=qos/math.exp(float(np.clip(sim.energy_spent_j[task]/max_e,0,1)))
        if not episode.finish_dag(reward_total=reward_sum):break
    return {'algorithm':behavior,'seed':seed,'tasks_scheduled':total_actions,
        'resource_utilization_task_counts':counts,**episode.report()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,default=DATASET_DIR/'sumo_trace.csv')
    p.add_argument('--start-sec',type=float,default=None)
    p.add_argument('--episode-index',type=int,default=0)
    args=p.parse_args()
    trace=MobilityTrace(args.trace,SUMO_DIR/'three_servers.json')
    start=args.start_sec if args.start_sec is not None else trace.times[0]
    results=[run(trace,b,4000,args.episode_index,start) for b in ('heft_online','earliest_finish','edf','lowest_energy')]
    out=DATASET_DIR/'heft_online_comparison.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'same_trace':str(args.trace),'start_sec':start,
                               'episode_index':args.episode_index,'split':'test',
                               'same_dag_seeds':True,'per_algorithm_release_time_may_differ':True,
                               'results':results},indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
