"""Live GUI overlay for 16 moving compute vehicles and 4 fixed Edge servers.

No graph snapshots are written at SUMO ticks; event/snapshot generation is in
muxgnn.mobility3.generate. The overlay is only a visual inspection tool.
"""
from __future__ import annotations
import os,sys,shutil,math,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def executable():
    name='sumo-gui.exe' if os.name=='nt' else 'sumo-gui'
    candidates=[shutil.which('sumo-gui')]
    home=os.environ.get('SUMO_HOME')
    if home:candidates.append(str(Path(home)/'bin'/name))
    if os.name=='nt':
        candidates.append(str(Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Eclipse'/'Sumo'/'bin'/name))
    for c in candidates:
        if c and Path(c).is_file():return c
    raise SystemExit('SUMO-GUI not found; set SUMO_HOME or add SUMO/bin to PATH')

def main():
    binary=executable()
    if 'SUMO_HOME' in os.environ:sys.path.insert(0,str(Path(os.environ['SUMO_HOME'])/'tools'))
    else:sys.path.insert(0,str(Path(binary).resolve().parent.parent/'tools'))
    import traci
    # Rebuild deterministic start locations for GUI even if a previous
    # capture used a different seed. Never change any computation settings.
    from prepare_rerouters import make
    make(seed=42,seconds=600.0)
    config=json.loads((ROOT/'three_servers.json').read_text(encoding='utf-8'))
    servers=config['servers'];v2v=float(config['v2v_range_m'])
    traci.start([binary,'-c',str(ROOT/'six_vehicles_three_servers.sumocfg'),'--no-step-log','true'])
    visible=set();colors={};previous=set();samples=0
    link_up_counts={'V2V':0,'V2I':0};link_down_counts={'V2V':0,'V2I':0}
    route_changes={f'V{k}':0 for k in range(16)};last_routes={}
    expected={f'V{k}' for k in range(16)};all_appeared=False
    try:
        try:
            view=traci.gui.getIDList()[0]
            traci.gui.setBoundary(view,0,0,1852.71,1146.27)
        except Exception:pass
        while traci.simulation.getTime()<600:
            traci.simulationStep();samples+=1
            ids=sorted(traci.vehicle.getIDList())
            actual=set(ids)
            if actual-expected:raise RuntimeError(f'Unexpected vehicle IDs: {sorted(actual-expected)}')
            if actual==expected:all_appeared=True
            elif all_appeared:raise RuntimeError(f'A vehicle disappeared from SUMO: {sorted(expected-actual)}')
            arrived=set(traci.simulation.getArrivedIDList())
            if arrived:raise RuntimeError(f'Vehicle arrived/was removed: {sorted(arrived)}')
            teleport=set(traci.simulation.getStartingTeleportIDList())
            if teleport:raise RuntimeError(f'Unexpected teleport started: {sorted(teleport)}')
            for vid in ids:
                route=tuple(traci.vehicle.getRoute(vid))
                if vid in last_routes and last_routes[vid] != route:
                    route_changes[vid]+=1
                    print(f't={traci.simulation.getTime():.2f}s {vid} new random destination (SUMO rerouter)')
                last_routes[vid]=route
            xy={v:traci.vehicle.getPosition(v) for v in ids}
            active={}
            for i,a in enumerate(ids):
                for b in ids[i+1:]:
                    if math.dist(xy[a],xy[b])<=v2v:active[('V2V',a,b)]=(xy[a],xy[b])
                for server in servers:
                    s=(float(server['x']),float(server['y']))
                    if math.dist(xy[a],s)<=server['range_m']:active[('V2I',a,server['id'])]=(xy[a],s)
            new=set(active)
            for k in visible-new:
                link_down_counts[k[0]]+=1
                traci.polygon.remove('link_'+'_'.join(k))
            for k,(p1,p2) in active.items():
                name='link_'+'_'.join(k)
                if k not in visible:
                    link_up_counts[k[0]]+=1
                    traci.polygon.add(name,[p1,p2],color=(240,157,40,235) if k[0]=='V2V' else (30,201,125,235),fill=False,polygonType=k[0],layer=160,lineWidth=3)
                else:traci.polygon.setShape(name,[p1,p2])
            visible=new
            if samples%100==0:
                print(f"t={traci.simulation.getTime():.1f}s vehicles={len(ids)} V2V={sum(k[0]=='V2V' for k in visible)} V2I={sum(k[0]=='V2I' for k in visible)}")
    finally:traci.close()
    if not all_appeared:raise RuntimeError('The sixteen vehicles never appeared simultaneously in SUMO')
    print('Observed reroutes per vehicle:',route_changes)
    print('Link UP counts:',link_up_counts,'Link DOWN counts:',link_down_counts)

if __name__=='__main__':main()
