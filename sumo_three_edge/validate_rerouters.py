"""Static mobility audit (SUMO runtime test still required for dynamical proof)."""
import hashlib,json,random,xml.etree.ElementTree as ET
from pathlib import Path
from prepare_rerouters import network_graph,tarjan,shortest_path,NET_FILE,ROUTE_FILE,REROUTER_FILE,AUDIT_FILE
HERE=Path(__file__).resolve().parent

def audit():
    adj,length,travel,coords=network_graph()
    core=tarjan(adj)[0]
    exits={e for e in core if adj[e]-core}
    routes=ET.parse(ROUTE_FILE).getroot()
    add=ET.parse(REROUTER_FILE).getroot()
    rr=add.findall('rerouter')
    mapping={x.get('edges'):x for x in rr}
    dest={t.get('id') for r in rr for t in r.findall('interval/destProbReroute')}
    expected={f'V{i}' for i in range(16)}
    vehicles={v.get('id') for v in routes.findall('vehicle')}
    assert vehicles==expected
    assert len(rr)==len(mapping)
    assert exits.issubset(mapping),f'exit guards missing: {sorted(exits-set(mapping))}'
    assert dest.issubset(core)
    assert len(dest)>=12
    for e,r in mapping.items():
        assert e in core
        opts=r.findall('interval/destProbReroute')
        assert len(opts)>=len(dest)-1
        assert all(t.get('id')!=e and t.get('id') in core for t in opts)
        assert all(t.get('probability')=='1' for t in opts)
    paths={r.get('id'):r.get('edges').split() for r in routes.findall('route')}
    assert len(paths)==16
    for r in routes.findall('route'):
        assert r.get('repeat') is None
        seq=r.get('edges').split()
        assert all(b in adj[a] for a,b in zip(seq,seq[1:]))
        assert all(e in core for e in seq)
        assert seq[-1] in dest
    for v in routes.findall('vehicle'):
        assert v.get('route') in paths and v.get('departEdge')=='0' and v.get('departPos')=='random_free'
    assert ET.parse(HERE/'six_vehicles_three_servers.sumocfg').getroot().find('.//additional-files').get('value').split(',')[-1]=='persistent_random_rerouters.add.xml'
    # A deterministic offline graph-level Monte Carlo smoke test; not SUMO movement.
    rng=random.Random(42)
    distinct={}
    for vid in sorted(expected):
        p=paths['seeded_path_'+vid][-1]
        edges=set(paths['seeded_path_'+vid]);history=[p]
        for _ in range(20):
            options=[t.get('id') for t in mapping[p].findall('interval/destProbReroute')]
            p2=rng.choice(options)
            path=shortest_path(adj,travel,p,p2,core)
            assert len(path)>=2 and all(b in adj[a] for a,b in zip(path,path[1:]))
            edges.update(path);p=p2;history.append(p)
        distinct[vid]={'destinations_seen':len(set(history)),'roads_visited':len(edges)}
    report={'valid':True,'passenger_edges':len(adj),'safe_core_edges':len(core),'unguarded_safe_core_exits':len(exits-set(mapping)),
            'rerouter_count':len(mapping),'unique_destinations':len(dest),'all_initial_routes_valid':True,
            'all_rerouter_destinations_valid':True,'seeded_offline_graph_smoke':distinct,
            'limitations':'graph-level proof only; install SUMO and run 600-second TraCI capture for dynamic survival'}
    (HERE/'rerouter_static_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
if __name__=='__main__':print(json.dumps(audit(),indent=2))
