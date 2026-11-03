"""Static 16-vehicle, 4-edge audit. No dynamic SUMO assertions are claimed."""
import csv,gzip,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from .settings import (PROJECT_ROOT,SUMO_DIR,NUM_VEHICLES,NUM_EDGE_SERVERS,DATASET_DIR,
                       TIME_SCALE_FACTOR,SUMO_STEP_LENGTH_SEC,SUMO_RANDOM_SEED)

def main():
    ref=PROJECT_ROOT/'data/muxgnn/legacy_native_dynamic_magnet_v4_physical_time'
    with (ref/'source_task_profiles.csv').open(encoding='utf-8') as f:profiles=list(csv.DictReader(f))
    with (ref/'dependency_edges.csv').open(encoding='utf-8') as f:deps=list(csv.DictReader(f))
    assert len(profiles)==60 and len(deps)==126
    assert NUM_VEHICLES==16 and NUM_EDGE_SERVERS==4
    net=ET.fromstring(gzip.decompress((SUMO_DIR/'osm.net.xml.gz').read_bytes()))
    edges={x.get('id') for x in net.findall('edge')}
    connections={(x.get('from'),x.get('to')) for x in net.findall('connection')}
    routes=ET.parse(SUMO_DIR/'six_vehicles.rou.xml').getroot()
    veh=routes.findall('vehicle');paths=routes.findall('route')
    assert len(veh)==len(paths)==NUM_VEHICLES
    assert {x.get('id') for x in veh}=={f'V{i}' for i in range(NUM_VEHICLES)}
    assert all(x.get('departEdge')=='0' and x.get('departPos')=='random_free' for x in veh)
    for route in paths:
        seq=route.get('edges').split()
        assert len(seq)>1 and all(x in edges for x in seq)
        assert all((a,b) in connections for a,b in zip(seq,seq[1:]))
        assert route.get('repeat') is None
    servers=json.loads((SUMO_DIR/'three_servers.json').read_text())
    assert len(servers['servers'])==NUM_EDGE_SERVERS
    assert {x['id'] for x in servers['servers']}=={f'EDGE{i}' for i in range(NUM_EDGE_SERVERS)}
    assert servers['v2v_range_m']==150 and servers['v2i_range_m']==300
    assert all(x['range_m']==300 and x['a7_cores']==x['a12_cores']==12 for x in servers['servers'])
    pois=ET.parse(SUMO_DIR/'three_edge_servers.add.xml').getroot().findall('poi')
    assert {p.get('id') for p in pois}=={f'EDGE{i}' for i in range(NUM_EDGE_SERVERS)}
    cfg=ET.parse(SUMO_DIR/'six_vehicles_three_servers.sumocfg').getroot()
    assert abs(float(cfg.find('time/step-length').get('value'))-SUMO_STEP_LENGTH_SEC)<1e-12
    assert int(cfg.find('random_number/seed').get('value'))==SUMO_RANDOM_SEED
    assert cfg.find('input/additional-files').get('value').split(',')[-1]=='persistent_random_rerouters.add.xml'
    with (DATASET_DIR/'derived_task_profiles_5x.csv').open(encoding='utf-8') as f:
        derived=list(csv.DictReader(f))
    assert len(derived)==60 and abs(max(float(x['deadline_sec']) for x in derived)-73.025)<1e-7
    report={'status':'STATIC_PASS','vehicles':NUM_VEHICLES,'servers':NUM_EDGE_SERVERS,
       'task_count':60,'dependency_count':126,'graph_nodes':80,'relation_types':5,
       'v2v_range_m':150,'v2i_range_m':300,'sumo_step_sec':SUMO_STEP_LENGTH_SEC,
       'sumo_seed':SUMO_RANDOM_SEED,'runtime_deadline_energy_factor':TIME_SCALE_FACTOR,
       'benchmark_sha256':hashlib.sha256((PROJECT_ROOT/'data/legacy/benchmark_profiles_a7_a12.csv').read_bytes()).hexdigest(),
       'dag_sha256':hashlib.sha256((PROJECT_ROOT/'data/legacy/dag_dataset_gang_a7_a12.csv').read_bytes()).hexdigest(),
       'sumo_execution_validated':False,'ppo_training_validated':False}
    DATASET_DIR.mkdir(parents=True,exist_ok=True)
    (DATASET_DIR/'STATIC_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
