"""Reproducible DAG-definition dataset, independent of unavailable SUMO runtime.

Actual event snapshots must later be captured from real SUMO trace via generate.
Definitions alone must NEVER be mistaken for motion snapshots or PPO results.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .multi_dag import episode_workloads,dag_summary,DAGS_PER_EPISODE
from .settings import DATASET_DIR


def export(count=24,output=DATASET_DIR):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    items=[]
    for epi in range(count):
        split='train' if epi%8<6 else 'val' if epi%8==6 else 'test'
        for dag_idx,g in enumerate(episode_workloads(split,epi)):
            info=dag_summary(g)
            path=output/'dag_definitions'/split/f'scenario_{epi:04d}'/f'dag_{dag_idx+1:02d}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            doc={**info, 'episode_index':epi,'dag_index':dag_idx+1,'split':split,
                 'dependencies':g['dependencies'],'topological_layers':g['layers'],
                 'roots':g['roots'],'sinks':g['sinks'],
                 'task_deadlines_sec':[float(p['deadline_sec']) for p in g['profiles']],
                 'task_runtime_a7_sec':[float(p['a7_time_sec']) for p in g['profiles']],
                 'task_runtime_a12_sec':[float(p['a12_time_sec']) for p in g['profiles']],
                 'task_energy_a7_j':[float(p['a7_energy_j']) for p in g['profiles']],
                 'task_energy_a12_j':[float(p['a12_energy_j']) for p in g['profiles']],
                 'task_gang_width':[int(p['gang_size']) for p in g['profiles']],
                 'origin':'derived synthetic topology from frozen v4 physical task profiles; not measured additional DAG workloads'}
            path.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
            items.append({**info,'episode_index':epi,'dag_index':dag_idx+1,'split':split,
                          'relative_path':str(path.relative_to(output))})
    assert len({x['fingerprint'] for x in items})==len(items)
    man={'version':'seven_dag_sequential_600s','scenarios':count,
         'dags_per_episode':DAGS_PER_EPISODE,'total_dag_definitions':len(items),
         'split_disjoint':True,'graphs':items,
         'requires_real_SUMO_trace_for_snapshots':True}
    (output/'dag_definitions_manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
    return man


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenarios',type=int,default=24)
    p.add_argument('--output',type=Path,default=DATASET_DIR)
    args=p.parse_args();print(export(args.scenarios,args.output)['total_dag_definitions'])
if __name__=='__main__':main()
