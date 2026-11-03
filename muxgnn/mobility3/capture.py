"""Collect exactly sixteen SUMO-actual vehicle traces via TraCI (no synthetic motion)."""
import argparse,csv,shutil,json,subprocess
from pathlib import Path
from .settings import SUMO_DIR,DATASET_DIR,SUMO_RANDOM_SEED,SUMO_STEP_LENGTH_SEC,NUM_VEHICLES

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=600.0)
    p.add_argument('--gui',action='store_true');p.add_argument('--output',type=Path,default=DATASET_DIR/'sumo_trace.csv')
    p.add_argument('--seed',type=int,default=SUMO_RANDOM_SEED);a=p.parse_args()
    try:import traci
    except ImportError as e:raise SystemExit('Install SUMO and pip install traci; SUMO_HOME must be set or sumo must be in PATH') from e
    binary='sumo-gui' if a.gui else 'sumo'
    if not shutil.which(binary):raise SystemExit(f'{binary} not found in PATH')
    cfg=SUMO_DIR/'six_vehicles_three_servers.sumocfg'
    # Generate native destination-rerouter XML for the requested seed; every
    # safe-network boundary guard and destination trigger is generated.
    subprocess.run([__import__('sys').executable,str(SUMO_DIR/'prepare_rerouters.py'),
                    '--seed',str(a.seed),'--seconds',str(a.seconds)],check=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    command=[binary,'-c',str(cfg),'--end',str(a.seconds),'--seed',str(a.seed),
             '--step-length',str(SUMO_STEP_LENGTH_SEC),'--no-step-log','true','--time-to-teleport','-1','--ignore-route-errors','false']
    traci.start(command)
    try: sumo_version = str(traci.getVersion())
    except Exception: sumo_version = 'unknown'
    n=0;first=None;last=None;failures=[];ever=set();all_ids={f'V{k}' for k in range(NUM_VEHICLES)}
    route_changes={v:0 for v in sorted(all_ids)};last_routes={};visited_edges={v:set() for v in all_ids}
    try:
        with a.output.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['time_sec','vehicle_id','x_m','y_m','speed_mps','angle_deg','road_id'])
            while traci.simulation.getTime()<a.seconds-1e-9:
                traci.simulationStep()
                t=float(traci.simulation.getTime());current=set(traci.vehicle.getIDList())
                if current-all_ids:failures.append({'time':t,'unexpected':sorted(current-all_ids)})
                if all_ids.issubset(current):
                    if first is None:first=t
                    for vid in sorted(all_ids):
                        route=tuple(traci.vehicle.getRoute(vid))
                        if vid in last_routes and route!=last_routes[vid]:route_changes[vid]+=1
                        last_routes[vid]=route
                        visited_edges[vid].add(traci.vehicle.getRoadID(vid))
                        x,y=traci.vehicle.getPosition(vid)
                        writer.writerow([f'{t:.6f}',vid,f'{x:.5f}',f'{y:.5f}',
                                         f'{traci.vehicle.getSpeed(vid):.6f}',
                                         f'{traci.vehicle.getAngle(vid):.6f}',traci.vehicle.getRoadID(vid)])
                    n+=1;last=t
                elif first is not None:failures.append({'time':t,'missing':sorted(all_ids-current)})
                ever.update(current)
                if hasattr(traci.simulation,'getStartingTeleportIDList'):
                    teleports=traci.simulation.getStartingTeleportIDList()
                    if teleports:failures.append({'time':t,'teleport':list(teleports)})
                arrivals=traci.simulation.getArrivedIDList()
                if arrivals:failures.append({'time':t,'unexpected_arrival':list(arrivals)})
    finally:traci.close()
    report={'valid':bool(n>=2 and not failures and ever==all_ids), 'samples_per_vehicle':n,
            'first_full_frame_sec':first,'last_frame_sec':last,'vehicles_seen':sorted(ever),
            'failures':failures[:40], 'failure_count':len(failures),
            'reroutes_observed_per_vehicle':route_changes,
            'unique_road_edges_seen_per_vehicle':{v:len(edges) for v,edges in visited_edges.items()},
            'route_generation_method':'SUMO native destProbReroute (seeded)', 'sumo_seed':a.seed,'sumo_step_length_sec':SUMO_STEP_LENGTH_SEC,'sumo_version':sumo_version,'config':str(cfg)}
    a.output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['valid']:raise SystemExit('SUMO trace validation failed: check .validation.json')

if __name__=='__main__':main()
