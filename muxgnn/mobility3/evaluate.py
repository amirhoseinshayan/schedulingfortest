"""Held-out PPO vs causal HEFT: identical seven graph seeds and SUMO trace."""
import argparse,json
from pathlib import Path
from stable_baselines3 import PPO
from .env import Mobility3Env
from .heft_online import run as run_heft
from .settings import DATASET_DIR,SUMO_DIR


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=Path('artifacts/mobility3/ppo_model.zip'))
    p.add_argument('--trace',type=Path,default=DATASET_DIR/'sumo_trace.csv')
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--output',type=Path,default=DATASET_DIR/'held_out_comparison.json')
    args=p.parse_args()
    if args.episodes<1:raise ValueError('--episodes must be >=1')
    env=Mobility3Env(trace_path=args.trace,seed=42,split='test',random_scenarios=False)
    model=PPO.load(args.model,env=env,device='cpu')
    results=[]
    for epi in range(args.episodes):
        obs,info=env.reset(options={'episode_index':epi,'start_sec':env.trace.times[0]})
        while True:
            action,_=model.predict(obs,deterministic=True)
            obs,reward,terminated,truncated,info=env.step(action)
            if terminated or truncated:break
        ppo_report=info['episode_report']
        heft_report=run_heft(env.trace,'heft_online',42+epi,epi,env.trace.times[0],split='test')
        if any(a['dag_seed']!=b['dag_seed'] for a,b in zip(
                ppo_report['completed_dags'],heft_report['completed_dags'])):
            raise AssertionError('Algorithms did not use identical DAG seed sequences')
        results.append({'episode_index':epi,'ppo':ppo_report,'heft':heft_report,
             'ppo_terminated':bool(terminated),'ppo_truncated':bool(truncated),
             'comparability':'identical DAGs, trace and seeded capacity; release timestamps depend on policy'})
        print('eval',epi,'PPO',ppo_report['completed_dag_count'],
              'HEFT',heft_report['completed_dag_count'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'split':'test','same_trace':str(args.trace),
          'results':results},indent=2),encoding='utf-8')
if __name__=='__main__':main()
