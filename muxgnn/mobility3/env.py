"""Causal event-driven PPO with preemptive migration and atomic (instant) transfers.

The SUMO trace is immutable and offline, the PPO scheduler is online. Frozen
five-relation muxGNN and the v4 QoS*exp(-energy/max_energy) formula are reused.
"""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from ..rl.adapter import HeteroMuxAdapter
from ..rl.config import (MUX_SCHEDULE_ACTIONS, MUX_TOTAL_ACTIONS, MUX_MAX_RESOURCES,
    MUX_MAX_TASKS, MUX_INVALID_ACTION_PENALTY, MUX_REWARD_PROPOSAL, MUX_QOS_FACTOR,
    MUX_MIGRATION_FEATURE_DIM)
from .scaled_source import load_scaled_task_graph
from .settings import DATASET_DIR,SUMO_DIR,MIN_TRACE_TAIL_SEC,NUM_VEHICLES,NUM_EDGE_SERVERS,DATASET_VERSION
from .trace import MobilityTrace
from .simulator import ThreeEdgeMobilitySimulator, MAX_MIGRATIONS_PER_TASK, WorkflowTimeLimitReached
from .episode import SevenDagEpisode
from .multi_dag import DAGS_PER_EPISODE, EPISODE_DURATION_SEC
from .builder import build_v4_snapshot,compute_v4_normalization


class Mobility3Env(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,trace_path=None,seed=42,split='train',random_scenarios=True):
        super().__init__()
        if MUX_MAX_RESOURCES != 20:
            raise RuntimeError('Set MUX_MOBILITY3=1 before importing PPO modules')
        self.trace=MobilityTrace(trace_path or DATASET_DIR/'sumo_trace.csv',SUMO_DIR/'three_servers.json')
        self.adapter=HeteroMuxAdapter()
        base=self.adapter.observation_space.spaces.copy()
        base['migration_features']=gym.spaces.Box(-np.inf,np.inf,
            shape=(MUX_MAX_TASKS,MUX_MAX_RESOURCES,2,MUX_MIGRATION_FEATURE_DIM),dtype=np.float32)
        base['migration_candidate_mask']=gym.spaces.MultiBinary((MUX_MAX_TASKS,MUX_MAX_RESOURCES,2))
        base['migration_task_mask']=gym.spaces.MultiBinary(MUX_MAX_TASKS)
        base['action_mask']=gym.spaces.MultiBinary(MUX_TOTAL_ACTIONS)
        self.observation_space=gym.spaces.Dict(base)
        self.action_space=gym.spaces.Discrete(MUX_TOTAL_ACTIONS)
        self.normalization=compute_v4_normalization()
        self.rng=np.random.default_rng(seed)
        self.seed_base=int(seed)
        self.split=split
        self.random_scenarios=bool(random_scenarios)
        self.episode=0
        src=load_scaled_task_graph()
        self.deadlines=np.asarray([p['deadline_sec'] for p in src['profiles']])
        self.max_energy=float(max(max(p['a7_energy_j'],p['a12_energy_j']) for p in src['profiles']))
        self.max_duration=float(max(max(p['a7_time_sec'],p['a12_time_sec']) for p in src['profiles']))
        self.sim=None
        self.event_log=[]
        self.rewarded_tasks=set()
        self.dag_reward=0.
        self.episode_runtime=None

    def _graph_obs(self,event_type):
        graph,meta=build_v4_snapshot(self.sim.state,len(self.event_log),{'trigger_event':event_type,'episode_start_sec':self.episode_runtime.start_sec,
            'episode_duration_sec':EPISODE_DURATION_SEC,'dag_index':self.episode_runtime.dag_index+1},
            self.split,self.normalization,workload=self.sim.workload)
        self.event_log.append({'event':event_type,**meta})
        obs=self.adapter.to_observation(graph)
        features=np.zeros((MUX_MAX_TASKS,MUX_MAX_RESOURCES,2,MUX_MIGRATION_FEATURE_DIM),dtype=np.float32)
        candidates=np.zeros((MUX_MAX_TASKS,MUX_MAX_RESOURCES,2),dtype=np.int8)
        task_mask=np.zeros(MUX_MAX_TASKS,dtype=np.int8)
        s=self.sim
        choices=s.valid_migrations()
        for c in choices:
            task,resource,proc=c.task_id,c.resource_slot,c.processor_index
            candidates[task,resource,proc]=1
            task_mask[task]=1
            old=int(s.state.assigned_resource_slot[task]);old_proc=int(s.state.assigned_processor_index[task])
            width=int(s.gang_size[task]);pool=s.state.free_a7 if proc==0 else s.state.free_a12
            total=s.total_a7 if proc==0 else s.total_a12
            # In this model transmission duration, bytes and radio power are NOT
            # synthesized: transfer is atomic and both radio metrics remain zero.
            features[task,resource,proc,:]=[
                c.progress,1.-c.progress,s.migration_count[task]/MAX_MIGRATIONS_PER_TASK,
                0.,old/(MUX_MAX_RESOURCES-1),float(old_proc==1),
                resource/(MUX_MAX_RESOURCES-1),float(proc==1),float(c.is_keep),
                min(1.,max(0.,(pool[resource]+(width if c.is_keep else 0))/max(total[resource],1))),
                0.,s._task_duration(task,old_proc)/max(s._task_duration(task,proc),1e-9),
                s._task_power(task,proc)/max(s._task_power(task,old_proc),1e-9),
                float(s.reachable(resource)),0.,0.,0.,
                c.remaining_duration_sec/max(self.max_duration,1e-9),
                (s.state.simulation_time_sec+c.remaining_duration_sec)/max(self.deadlines[task],1e-9),
                0.,
            ]
        obs['migration_features']=features
        obs['migration_candidate_mask']=candidates
        obs['migration_task_mask']=task_mask
        obs['action_mask']=np.concatenate([obs['action_mask'],candidates.reshape(-1)]).astype(np.int8)
        if not self.observation_space.contains(obs):
            raise AssertionError('PPO migration observation violates action/feature contract')
        self._last_observation=obs
        return obs

    def _sync_dag(self):
        """New DAG, new task deadlines; no PPO or physical SUMO reset."""
        self.sim=self.episode_runtime.sim
        self.deadlines=np.asarray([p['deadline_sec'] for p in self.sim.profiles],dtype=float)
        self.max_energy=float(max(max(p['a7_energy_j'],p['a12_energy_j']) for p in self.sim.profiles))
        self.max_duration=float(max(max(p['a7_time_sec'],p['a12_time_sec']) for p in self.sim.profiles))
        self.rewarded_tasks=set()
        self.dag_reward=0.

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng=np.random.default_rng(seed)
            self.seed_base=int(seed)
        self.episode+=1
        opt=options or {}
        epi=int(opt.get('episode_index',self.episode-1))
        start=float(opt.get('start_sec',self.trace.times[0]))
        self.episode_runtime=SevenDagEpisode(self.trace,self.split,epi,start,
            seed=self.seed_base+epi,behavior_policy='earliest_finish')
        self._sync_dag()
        self.event_log=[]
        self._graph_obs('initial_ready_tasks')
        if not np.any(self._last_observation['action_mask']):
            _,error,finished=self._advance_until_actionable()
            if error or finished:
                raise RuntimeError('Episode cannot start with an actionable task: '+str(error))
        return self._last_observation,self._info()

    def _info(self):
        s=self.sim
        e=self.episode_runtime
        return {'simulation_time_sec':float(s.state.simulation_time_sec),
                'sumo_time_sec':float(s.state.physical_time_sec),
                'global_episode_elapsed_sec':float(s.state.physical_time_sec-e.start_sec),
                'episode_limit_sec':EPISODE_DURATION_SEC,
                'dag_index':int(e.dag_index+1), 'dag_seed':int(s.workload['dag_seed']),
                'dag_critical_path_sec':float(s.workload['critical_path_sec']),
                'dag_deadline_sec':float(s.workload['workflow_deadline_sec']),
                'num_dags_completed':len(e.completed_dags),
                'dag_history':list(e.completed_dags),
                'num_completed_tasks':int(sum(s.state.completed)),
                'num_buffered_outputs':int(sum(s.awaiting_delivery)),
                'total_migrations':int(sum(s.migration_count)),
                'total_compute_energy_j':float(e.energy_episode_j+sum(s.energy_spent_j)),
                'dag_energy_j':float(sum(s.energy_spent_j)),
                'migration_transfer_delay_sec':0.,'migration_transfer_energy_j':0.,
                'makespan_sec':float(max([0.,*s.delivery_time_sec])),
                'num_edge_servers':NUM_EDGE_SERVERS,'num_vehicles':NUM_VEHICLES,
                'dataset_version':DATASET_VERSION,'time_scale_factor':5.0}

    @staticmethod
    def _qos(finish,deadline):
        if deadline<=0:return 0.
        if finish<=deadline:return 1.
        upper=MUX_QOS_FACTOR*deadline
        if finish>=upper:return 0.
        return float((upper-finish)/max(upper-deadline,1e-9))

    def _reward_delivered(self,tasks):
        reward=0.
        for task in tasks:
            if task in self.rewarded_tasks:raise RuntimeError('Task output delivered twice')
            self.rewarded_tasks.add(task)
            finish=float(self.sim.delivery_time_sec[task])
            qos=self._qos(finish,float(self.deadlines[task]))
            reward+=qos/math.exp(float(np.clip(self.sim.energy_spent_j[task]/self.max_energy,0,1)))
        return float(reward)

    def _advance_until_actionable(self):
        """Advance until valid PPO decision, next DAG, completion, or time cutoff.

        Returns (reward, error, naturally_terminated). No time-limit event
        is treated as successful completion; no old-DAG events cross boundaries.
        """
        reward=0.
        while True:
            s=self.sim
            if bool(np.all(s.state.completed)):
                has_next=self.episode_runtime.finish_dag(reward_total=self.dag_reward)
                if not has_next:
                    if len(self.episode_runtime.completed_dags)!=DAGS_PER_EPISODE:
                        return reward,'600_second_time_limit',False
                    return reward,None,True
                self._sync_dag()
                self._graph_obs('new_dag_released')
                continue
            if s._valid_actions() or s.valid_migrations():
                return reward,None,False
            try:
                event=s.advance_one_event()
            except WorkflowTimeLimitReached as exc:
                self.episode_runtime.record_incomplete(str(exc))
                return reward,'600_second_time_limit',False
            except RuntimeError as exc:
                self.episode_runtime.record_incomplete(str(exc))
                return reward,str(exc),False
            gained=self._reward_delivered(event['completed_tasks'])
            reward+=gained
            self.dag_reward+=gained
            self._graph_obs(event['trigger_event'])

    def step(self,action):
        s=self.sim
        action=int(action)
        if action<0 or action>=MUX_TOTAL_ACTIONS or not self._last_observation['action_mask'][action]:
            return self._last_observation,-float(MUX_INVALID_ACTION_PENALTY),False,False,{**self._info(),'invalid_action':True}
        if MUX_REWARD_PROPOSAL!='B':
            raise RuntimeError('Original QoS*exp(-normalized energy) reward contract changed')
        choice_index=action if action<MUX_SCHEDULE_ACTIONS else action-MUX_SCHEDULE_ACTIONS
        task=choice_index//(MUX_MAX_RESOURCES*2)
        resource=(choice_index//2)%MUX_MAX_RESOURCES
        processor=choice_index%2
        if action<MUX_SCHEDULE_ACTIONS:
            choices=[a for a in s._valid_actions() if (a.task_id,a.resource_slot,a.processor_index)==(task,resource,processor)]
            if not choices:raise RuntimeError('PPO schedule mask out of sync with simulator')
            s.state.last_link_events=[]
            s._start_task(choices[0])
            self._graph_obs('task_started')
            detail={'action_type':'schedule','task_id':task,'resource_slot':resource,'processor':processor}
        else:
            choices=[c for c in s.valid_migrations() if (c.task_id,c.resource_slot,c.processor_index)==(task,resource,processor)]
            if not choices:raise RuntimeError('PPO migration mask out of sync with simulator')
            detail=s.apply_migration(choices[0])
            self._graph_obs('migration_keep' if choices[0].is_keep else 'migration')
            detail={'action_type':'migration_keep' if choices[0].is_keep else 'migration',**detail}
        gained,error,finished=self._advance_until_actionable()
        info={**self._info(),**detail}
        if error:
            info['truncated_reason']=error
            info['episode_report']=self.episode_runtime.report()
            return self._last_observation,float(gained),False,True,info
        if finished:
            info['episode_report']=self.episode_runtime.report()
        return self._last_observation,float(gained),finished,False,info
