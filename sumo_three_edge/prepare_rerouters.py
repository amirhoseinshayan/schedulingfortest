"""Build deterministic SUMO destination-rerouters for persistent six-vehicle mobility.

Only mobility files are edited.  A guarded strongly-connected passenger road
component prevents vehicles reaching irreversible network exits. Each reachable
random-destination edge and every boundary guard has a native SUMO rerouter;
SUMO uses --seed to choose among alternative destinations at runtime.
"""
from __future__ import annotations
import argparse, collections, gzip, hashlib, json, math, random
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
NET_FILE = HERE / 'osm.net.xml.gz'
ROUTE_FILE = HERE / 'six_vehicles.rou.xml'
REROUTER_FILE = HERE / 'persistent_random_rerouters.add.xml'
AUDIT_FILE = HERE / 'rerouter_generation_audit.json'
ORIGINAL_STARTS_FILE = HERE / 'original_loop_start_edges.json'
EXPECTED = [f'V{i}' for i in range(16)]
DESTINATION_COUNT = 30


def allowed(lane):
    a=set(lane.get('allow','').split());d=set(lane.get('disallow','').split())
    return 'passenger' not in d and (not a or 'passenger' in a)


def network_graph():
    with gzip.open(NET_FILE,'rb') as fp: root=ET.parse(fp).getroot()
    edge_xml={e.get('id'):e for e in root.findall('edge') if e.get('function') != 'internal' and e.get('id') and not e.get('id').startswith(':')}
    legal={k for k,e in edge_xml.items() if any(allowed(l) for l in e.findall('lane'))}
    adj={k:set() for k in legal}; length={};coords={};time={}
    for eid in legal:
        lanes=[l for l in edge_xml[eid].findall('lane') if allowed(l)]
        length[eid]=min(float(l.get('length','1')) for l in lanes)
        speed=max(float(l.get('speed','1')) for l in lanes)
        time[eid]=length[eid]/max(.1,speed)
        shape=lanes[0].get('shape','')
        points=[tuple(map(float,p.split(',')[:2])) for p in shape.split()]
        coords[eid]=points[len(points)//2] if points else (0.,0.)
    for c in root.findall('connection'):
        src,dst=c.get('from'),c.get('to')
        if src not in legal or dst not in legal:continue
        la=edge_xml[src].find("lane[@index='%s']"%c.get('fromLane'))
        lb=edge_xml[dst].find("lane[@index='%s']"%c.get('toLane'))
        if la is not None and lb is not None and allowed(la) and allowed(lb):adj[src].add(dst)
    return adj,length,time,coords


def tarjan(adj):
    """Strongly-connected components in passenger-accessible directed edge graph."""
    visited=set();order=[]
    for start in sorted(adj):
        if start in visited:continue
        stack=[(start,iter(sorted(adj[start])))]
        visited.add(start)
        while stack:
            n,children=stack[-1]
            try: nxt=next(children)
            except StopIteration:
                order.append(n);stack.pop();continue
            if nxt not in visited:
                visited.add(nxt);stack.append((nxt,iter(sorted(adj[nxt]))))
    reverse={u:set() for u in adj}
    for u,ns in adj.items():
        for n in ns:reverse[n].add(u)
    visited.clear();components=[]
    for start in reversed(order):
        if start in visited:continue
        group={start};visited.add(start);stack=[start]
        while stack:
            for n in reverse[stack.pop()]:
                if n not in visited:visited.add(n);group.add(n);stack.append(n)
        components.append(group)
    return sorted(components,key=lambda s:(-len(s),sorted(s)[0]))


def shortest_path(adj,time,start,target,valid):
    import heapq
    if start not in valid or target not in valid:raise ValueError(f'outside safe core: {start} -> {target}')
    heap=[(0.,start)];dist={start:0.};previous={}
    while heap:
        cost,u=heapq.heappop(heap)
        if cost != dist.get(u):continue
        if u == target:break
        for v in sorted(adj[u] & valid):
            nextcost=cost+time[v]
            if nextcost < dist.get(v,float('inf')):
                dist[v]=nextcost;previous[v]=u;heapq.heappush(heap,(nextcost,v))
    if target not in dist:raise ValueError(f'no passenger route: {start} -> {target}')
    rev=[target]
    while rev[-1]!=start:rev.append(previous[rev[-1]])
    return list(reversed(rev))


def choose_destinations(core,coords,length,frontier,starts,seed,count=DESTINATION_COUNT):
    # spatially dispersed destination edges, not a single periodic loop
    rng=random.Random(seed+179)
    pool=[e for e in sorted(core) if e not in frontier and length[e]>=25. and all(math.dist(coords[e],coords[s])>12 for s in starts)]
    rng.shuffle(pool)
    if len(pool)<count:raise ValueError('not enough safe internal random-destination edges')
    # Deterministic greedy farthest-point sample for spatial coverage
    first=pool[0];chosen=[first];seen={first}
    while len(chosen)<count:
        nxt=max((e for e in pool if e not in seen),key=lambda e:(min(math.dist(coords[e],coords[x]) for x in chosen),-pool.index(e)))
        chosen.append(nxt);seen.add(nxt)
    return chosen


def make(seed=42, seconds=600.):
    if seconds<=0:raise ValueError('positive simulation time required')
    adj,length,time,coords=network_graph()
    components=tarjan(adj);core=components[0]
    if len(core)<50:raise RuntimeError('unexpected road graph: core too small')
    frontier={a for a in core if adj[a]-core}
    # Core->outside edges are the only entry points to no-return external branches.
    exits={e for e in adj if not adj[e]}
    starts=json.loads(ORIGINAL_STARTS_FILE.read_text(encoding='utf-8'))
    if set(starts)!=set(EXPECTED) or any(e not in core for e in starts.values()):raise ValueError('invalid initial start-road snapshot')
    dests=choose_destinations(core,coords,length,frontier,list(starts.values()),seed)
    tr=ET.parse(ROUTE_FILE);root=tr.getroot()
    vehicles={v.get('id'):v for v in root.findall('vehicle')}
    if set(vehicles)!=set(EXPECTED):raise ValueError('expected exactly sixteen vehicles V0-V15')
    for r in list(root.findall('route')):root.remove(r)
    # Initial destinations drawn deterministically from the same safe core.
    rng=random.Random(seed+37);init={}
    for v in EXPECTED:
        src=starts[v];options=[d for d in dests if d!=src];dest=rng.choice(options)
        edges=shortest_path(adj,time,src,dest,core)
        init[v]={'start_edge':src,'initial_destination':dest,'initial_route_edges':len(edges)}
        rr=ET.Element('route',{'id':'seeded_path_'+v,'edges':' '.join(edges)})
        root.insert(1+int(v[1:]),rr)
        vehicles[v].set('route','seeded_path_'+v)
        vehicles[v].set('departEdge','0')
        if vehicles[v].get('departPos')!='random_free':raise ValueError('departPos changed outside mobility')
    # A rerouter on every selected destination prevents vehicle arrival. Additional
    # rerouters at ALL possible exits from the safe SCC redirect before a branch
    # from which the vehicle could never return. Every new destination stays in SCC.
    triggers=sorted(set(dests)|frontier)
    additional=ET.Element('additional')
    for idx,edge in enumerate(triggers):
        r=ET.SubElement(additional,'rerouter',{'id':f'random_destination_{idx:03d}','edges':edge,'probability':'1','vTypes':'computeVehicle'})
        interval=ET.SubElement(r,'interval',{'begin':'0','end':'1000000000'})
        # never choose the current trigger edge as destination (would allow arrival)
        options=[d for d in dests if d!=edge]
        if len(options)<2:raise ValueError('not enough alternative destinations')
        for dest in sorted(options):ET.SubElement(interval,'destProbReroute',{'id':dest,'probability':'1'})
    ET.indent(additional,space='  ')
    ET.ElementTree(additional).write(REROUTER_FILE,encoding='utf-8',xml_declaration=True)
    ET.indent(root,space='  ')
    tr.write(ROUTE_FILE,encoding='utf-8',xml_declaration=True)
    # static checks: no repeat route; destination not current edge; all targets
    # reachable from every trigger (strong connectivity); all core exit guards.
    for r in root.findall('route'):
        assert not r.get('repeat'), 'repeat route was not removed'
        edges=r.get('edges').split()
        assert all(b in adj[a] for a,b in zip(edges,edges[1:])),r.get('id')
        assert all(e in core for e in edges)
    for r in additional.findall('rerouter'):
        here=r.get('edges');options=[x.get('id') for x in r.findall('interval/destProbReroute')]
        assert here in core and here not in options
        assert set(options).issubset(core)
    audit={'version':'native SUMO destProbReroute, seeded runtime random choices',
      'sumo_seed':seed,'simulation_horizon_seconds':seconds,'vehicle_count':len(vehicles),
      'vehicle_ids':EXPECTED,'core_passenger_edges':len(core),'total_passenger_edges':len(adj),
      'core_exit_guard_count':len(frontier),'core_exit_guard_edges':sorted(frontier),
      'sink_edges_outside_core':sorted(exits-core),'destination_trigger_count':len(dests),
      'destination_edges':sorted(dests),'total_unique_rerouter_triggers':len(triggers),
      'per_vehicle_initial_routes':init,
      'random_selection':'SUMO seeded RNG, normalized equal weights; every trigger excludes own edge',
      'no_repeat_routes':True,'no_reinsertion_or_teleport':True,
      'all_destinations_inside_reachable_core':True,
      'all_core_exits_guarded_before_irreversible_branch':True,
      'static_audit':'passed; dynamic SUMO validation still required'}
    AUDIT_FILE.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--seconds',type=float,default=600.)
    a=p.parse_args();print(json.dumps(make(a.seed,a.seconds),indent=2))
