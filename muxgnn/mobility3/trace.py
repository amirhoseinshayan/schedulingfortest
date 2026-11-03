from __future__ import annotations
import bisect,csv,math,collections,json
from pathlib import Path
from typing import Dict
from .settings import NUM_VEHICLES,V2V_RANGE_M,V2I_RANGE_M

class MobilityTrace:
    """Causal zero-order hold; never pass future trajectory to an agent.

    The entire trace is known to scenario generation, but at(t) returns only
    recorded samples with timestamps <= t. Link events are precomputed offline.
    """
    def __init__(self, path, servers_path):
        self.path=Path(path)
        info=json.loads(Path(servers_path).read_text(encoding='utf-8'))
        self.servers=tuple(info['servers'])
        assert len(self.servers)==4
        self.v2v_range=float(info['v2v_range_m']);self.v2i_range=float(info['v2i_range_m'])
        records=collections.defaultdict(dict)
        with self.path.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                vid=row['vehicle_id']
                if vid not in [f'V{i}' for i in range(NUM_VEHICLES)]:continue
                t=round(float(row['time_sec']),7)
                if vid in records[t]: raise ValueError('duplicate vehicle sample')
                records[t][vid]={'x':float(row['x_m']),'y':float(row['y_m']),
                                 'speed':float(row['speed_mps']),'angle':float(row['angle_deg'])}
        self.times=sorted(records)
        if len(self.times)<2:raise ValueError('need 2+ movement samples')
        for t in self.times:
            if len(records[t])!=NUM_VEHICLES:
                raise ValueError(f'All {NUM_VEHICLES} vehicles must be present at {t}: got {list(records[t])}')
        self.frames=[records[t] for t in self.times]
        self.events=[];previous=None
        for t,frame in zip(self.times,self.frames):
            state=self._links(frame)
            if previous is not None:
                changes=[]
                for link in sorted(state.symmetric_difference(previous)):
                    changes.append({'event':'LINK_UP' if link in state else 'LINK_DOWN',
                                    'link_type':link[0], 'source':link[1],'target':link[2]})
                if changes:self.events.append((t,changes))
            previous=state
        self.end=self.times[-1]

    def _links(self,frame):
        active=set()
        for i in range(NUM_VEHICLES):
            a=frame[f'V{i}']
            for j in range(i+1,NUM_VEHICLES):
                b=frame[f'V{j}']
                if math.hypot(a['x']-b['x'],a['y']-b['y'])<=self.v2v_range:
                    active.add(('V2V',f'V{i}',f'V{j}'))
            for si,s in enumerate(self.servers):
                if math.hypot(a['x']-s['x'],a['y']-s['y'])<=float(s['range_m']):
                    active.add(('V2I',f'V{i}',f'EDGE{si}'))
        return active

    def at(self,t):
        idx=bisect.bisect_right(self.times,float(t)+1e-9)-1
        if idx<0:raise ValueError('trace has no sample at/before request time')
        frame=self.frames[idx]
        links=self._links(frame)
        connected_v2v={i for i in range(1,NUM_VEHICLES) if ('V2V','V0',f'V{i}') in links}
        connected_edge={i for i in range(len(self.servers)) if ('V2I','V0',f'EDGE{i}') in links}
        return {'sample_time_sec':self.times[idx], 'vehicles':frame,
                'links':links,'v2v_candidates':connected_v2v,'edge_candidates':connected_edge}
