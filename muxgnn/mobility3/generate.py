"""Seven new DAGs per 600 s SUMO trace episode: offline MuxGNN snapshots.

Only one active DAG at any instant; next DAG is released precisely at previous
DAG completion. Unfinished workflows remain labeled as incomplete (no fake
success); raw reference v4 benchmark/DAG files and SUMO trace stay immutable.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

import numpy as np

from ..graph.v4_source import sha256
from .builder import build_v4_snapshot, compute_v4_normalization
from .episode import SevenDagEpisode
from .multi_dag import DAGS_PER_EPISODE, EPISODE_DURATION_SEC, dag_summary
from .simulator import WorkflowTimeLimitReached
from .trace import MobilityTrace
from .scaled_source import load_scaled_task_graph
from .settings import DATASET_DIR,SUMO_DIR,DATASET_VERSION,TIME_SCALE_FACTOR,SUMO_STEP_LENGTH_SEC,SUMO_RANDOM_SEED,NUM_VEHICLES,NUM_EDGE_SERVERS


def generate(trace_path, output, scenarios=24, start_sec=None):
    import torch
    trace=MobilityTrace(trace_path,SUMO_DIR/'three_servers.json')
    start=float(trace.times[0] if start_sec is None else start_sec)
    if trace.end+1e-8 < start+EPISODE_DURATION_SEC:
        raise ValueError('Capture a complete 600s SUMO trace from the requested start time')
    source=load_scaled_task_graph(); norm=compute_v4_normalization()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    report=[]; split_map={'train':[],'val':[],'test':[]}
    for i in range(scenarios):
        split='train' if i%8 <6 else 'val' if i%8==6 else 'test'
        policy=('earliest_finish','edf','lowest_energy')[i%3]
        scenario=f'scenario_{i:04d}'
        episode=SevenDagEpisode(trace,split,i,start,seed=4000+i,behavior_policy=policy)
        folder=output/split/scenario; folder.mkdir(parents=True,exist_ok=True)
        rows=[]
        for dag_index in range(DAGS_PER_EPISODE):
            sim=episode.sim
            w=episode.workloads[dag_index]
            snapdir=folder/f'dag_{dag_index+1:02d}'/'snapshots'
            snapdir.mkdir(parents=True,exist_ok=True)
            truncated=False
            try:
                for epoch in sim.iter_decision_epochs():
                    graph,meta=build_v4_snapshot(sim.state,epoch.index,{**epoch.to_metadata(),'episode_start_sec':episode.start_sec,
                        'episode_duration_sec':EPISODE_DURATION_SEC,'dag_index':dag_index+1},split,norm,workload=w)
                    rel=str((snapdir/f'snapshot_{epoch.index:05d}.pt').relative_to(folder))
                    torch.save(graph,folder/rel)
                    rows.append({'dag_index':dag_index+1,'dag_seed':w['dag_seed'],
                        'dag_fingerprint':w['dag_fingerprint'],'epoch':epoch.index,
                        'workflow_relative_sec':epoch.simulation_time_sec,
                        'sumo_time_sec':sim.state.physical_time_sec,
                        'event_type':epoch.trigger_event,'started_task':epoch.started_task,
                        'completed_tasks':json.dumps(list(epoch.completed_tasks)),
                        'newly_ready_tasks':json.dumps(list(epoch.newly_ready_tasks)),
                        'link_events_json':json.dumps(meta['link_events'],separators=(',',':')),
                        'local_edges':meta['num_local_edges'],'v2v_edges':meta['num_v2v_edges'],
                        'v2i_edges':meta['num_v2i_edges'],'snapshot_file':rel})
            except WorkflowTimeLimitReached:
                truncated=True
            except RuntimeError as exc:
                episode.record_incomplete(str(exc))
                break
            if truncated or not bool(np.all(sim.state.completed)):
                episode.record_incomplete('600_second_time_limit' if truncated else 'incomplete')
                break
            if not episode.finish_dag():
                break
        if rows:
            with (folder/'event_log.csv').open('w',newline='',encoding='utf-8') as f:
                dw=csv.DictWriter(f,fieldnames=list(rows[0]));dw.writeheader();dw.writerows(rows)
        item={'scenario':scenario,'split':split,'seed':4000+i,'behavior_policy':policy,
              'start_sumo_sec':start,'last_sumo_sec':episode.sim.state.physical_time_sec,
              'num_event_snapshots':len(rows),**episode.report()}
        (folder/'scenario.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
        report.append(item);split_map[split].append(scenario)
        print(scenario,split,'DAGs',len(episode.completed_dags),'snapshots',len(rows))
    manifest={'version':DATASET_VERSION,'source_v4_immutable':True,
      'frozen_source_dag_sha256':sha256(Path(__file__).resolve().parents[2]/'data/legacy/dag_dataset_gang_a7_a12.csv'),
      'frozen_source_profiles_sha256':sha256(Path(__file__).resolve().parents[2]/'data/legacy/benchmark_profiles_a7_a12.csv'),
      'source_trace_sha256':sha256(Path(trace_path)),'sumo_random_seed':SUMO_RANDOM_SEED,
      'sumo_step_length_sec':SUMO_STEP_LENGTH_SEC,'synthetic_runtime_deadline_energy_factor':TIME_SCALE_FACTOR,
      'num_tasks_per_dag':len(source['profiles']),'num_dependencies_per_dag':len(source['dependencies']),
      'num_dags_target':DAGS_PER_EPISODE,'episode_max_duration_sec':EPISODE_DURATION_SEC,
      'num_vehicles':NUM_VEHICLES,'num_edge_servers':NUM_EDGE_SERVERS,
      'num_nodes_per_snapshot':len(source['profiles'])+NUM_VEHICLES+NUM_EDGE_SERVERS,
      'num_relation_types':5,'dag_seed_policy':'disjoint split ranges; unique per episode and DAG',
      'dag_deadline_policy':'independently recompute CP_i and deadline_i = 1.15 * CP_i',
      'dag_release_policy':'release immediately after full previous DAG output delivery',
      'energy_policy':'per-DAG independent, episode cumulative; no battery constraint',
      'split_independent_dags':True,'offline_sumo_trace':True,
      'snapshot_policy':'initial, task, or link decision events; never per-SUMO-step full graph',
      'unfinished_policy':'explicitly report censored work; do not relabel as successful',
      'splits':split_map,'scenarios':report}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',type=Path,default=DATASET_DIR/'sumo_trace.csv')
    ap.add_argument('--scenarios',type=int,default=24)
    ap.add_argument('--output',type=Path,default=DATASET_DIR)
    ap.add_argument('--start-sec',type=float,default=None)
    args=ap.parse_args()
    generate(args.trace,args.output,args.scenarios,args.start_sec)
    print('DONE',args.output)

if __name__=='__main__':main()
