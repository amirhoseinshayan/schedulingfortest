"""Dependency-light audit: check each scaled record against immutable v4 CSV."""
from __future__ import annotations
import csv,hashlib,json,math,xml.etree.ElementTree as ET
from pathlib import Path
from .settings import PROJECT_ROOT,DATASET_DIR,SUMO_DIR,TIME_SCALE_FACTOR,SUMO_STEP_LENGTH_SEC,SUMO_RANDOM_SEED,NUM_VEHICLES,NUM_EDGE_SERVERS


def _read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ref=PROJECT_ROOT/'data/muxgnn/legacy_native_dynamic_magnet_v4_physical_time/source_task_profiles.csv'
    derived=DATASET_DIR/'derived_task_profiles_5x.csv'
    rows_ref,rows_new=_read_csv(ref),_read_csv(derived)
    assert len(rows_ref)==len(rows_new)==60
    k=TIME_SCALE_FACTOR
    multiplied=['a7_execution_time_ms','a12_execution_time_ms','a7_time_sec','a12_time_sec',
      'a7_gang_time_sec','a12_gang_time_sec','a7_energy_j','a12_energy_j',
      'deadline_sec','dependency_earliest_ready_sec','dependency_earliest_finish_sec',
      'deadline_slack_from_dependency_earliest_finish_sec']
    unchanged=['task_id','benchmark_name','topological_layer','gang_size','release_time_sec',
      'period_placeholder','a7_power_w','a12_power_w','legacy_deadline_canonical',
      'legacy_deadline_physical_sec']
    count=0
    for a,b in zip(rows_ref,rows_new):
      for key in multiplied:
        assert math.isclose(float(b[key]),k*float(a[key]),rel_tol=1e-11,abs_tol=1e-9),(a['task_id'],key)
        count+=1
      for key in unchanged:
        assert b[key]==a[key],(a['task_id'],key)
      for arch in ('a7','a12'):
        assert math.isclose(float(b[arch+'_energy_j']),float(b[arch+'_time_sec'])*float(b[arch+'_power_w']),rel_tol=1e-12)
      assert math.isclose(float(b['deadline_sec']),1.15*float(b['dependency_earliest_finish_sec']),rel_tol=1e-10)
      assert int(b['gang_size'])<=6
    source_hash=json.loads((DATASET_DIR/'SCALING_VALIDATION.json').read_text())
    assert _sha(ref)==source_hash['source_files_sha256']['v4_profiles']
    assert _sha(derived)==source_hash['derived_profiles_sha256']
    ref_dep=PROJECT_ROOT/'data/muxgnn/legacy_native_dynamic_magnet_v4_physical_time/dependency_edges.csv'
    assert _sha(DATASET_DIR/'dependency_edges.csv')==_sha(ref_dep)
    cfg=ET.parse(SUMO_DIR/'six_vehicles_three_servers.sumocfg').getroot()
    assert float(cfg.find('time/step-length').get('value'))==SUMO_STEP_LENGTH_SEC
    assert int(cfg.find('random_number/seed').get('value'))==SUMO_RANDOM_SEED
    route=ET.parse(SUMO_DIR/'six_vehicles.rou.xml').getroot()
    vehicles=route.findall('vehicle')
    assert len(vehicles)==NUM_VEHICLES and len(route.findall('route'))==NUM_VEHICLES
    assert all(v.get('departEdge')=='0' and v.get('departPos')=='random_free' for v in vehicles)
    result={'status':'PASS','scaled_fields_checked':count,'tasks_checked':60,
            'time_factor':k,'energy_factor':k,'power_factor':1.0,
            'sumo_step_length_sec':SUMO_STEP_LENGTH_SEC,'sumo_seed':SUMO_RANDOM_SEED,
            'source_file_sha256':_sha(ref),'scaled_profile_sha256':_sha(derived),
            'note':'STATIC math/XML audit; SUMO and PPO dynamic tests are separate'}
    (DATASET_DIR/'SCALING_REVALIDATED.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
