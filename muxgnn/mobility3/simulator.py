from __future__ import annotations

"""Trace-driven task scheduling with causal, zero-transfer-time migration.

Frozen v4 profile, gang semantics, and power remain unchanged. Transfer is
atomic and free *only if both endpoints can reach V0 at decision time*. A lost
link never causes teleportation of a checkpoint: computation keeps running and
remote results are buffered until the source reconnects.
"""
import heapq
import math
from dataclasses import dataclass

import numpy as np
from ..graph.v4_simulation import LegacyNativeV4ScenarioSimulator, V4CompletionEvent, V4ReferenceAction
from ..graph.v4_settings import V4_EDGE_MIN_FREE_CORES, V4_REACHABLE_V2V_MIN_FREE_CORES, V4_EVENT_TIME_EPSILON_SEC
from .settings import NUM_VEHICLES, NUM_EDGE_SERVERS, EDGE_A7_CORES, EDGE_A12_CORES, VEHICLE_A7_CORES, VEHICLE_A12_CORES
from .scaled_source import load_scaled_task_graph

EPS = 1e-9
class WorkflowTimeLimitReached(RuntimeError):
    """The current SUMO episode reached its fixed 600-second boundary."""

MAX_MIGRATIONS_PER_TASK = 3
MIGRATION_CHECK_FRACTION = 0.5
MIN_CHECK_INTERVAL = 0.05  # cannot exceed SUMO's time resolution


@dataclass(frozen=True)
class MigrationChoice:
    task_id: int
    resource_slot: int
    processor_index: int
    is_keep: bool
    progress: float
    remaining_duration_sec: float
    expected_remaining_energy_j: float


class ThreeEdgeMobilitySimulator(LegacyNativeV4ScenarioSimulator):
    """Event-driven CPU scheduling plus mobility link and migration events."""

    def __init__(self, trace, scenario_id=0, seed=4000, behavior_policy='earliest_finish', start_sec=0.0, workload=None, capacity_override=None, episode_end_sec=None):
        super().__init__(scenario_id, seed, behavior_policy)
        self.workload = workload if workload is not None else load_scaled_task_graph()
        self.profiles = list(self.workload['profiles'])
        self.dag = self.workload['dag']
        self.predecessors = [[] for _ in range(self.num_tasks)]
        self.successors = [[] for _ in range(self.num_tasks)]
        for u,v in self.workload['dependencies']:
            self.predecessors[int(v)].append(int(u))
            self.successors[int(u)].append(int(v))
        self.release_time_sec = np.asarray([p['release_time_sec'] for p in self.profiles], dtype=np.float64)
        self.state.release_time_sec = self.release_time_sec.copy()
        self.state.ready[:] = False
        self.state.ready_time_sec[:] = -1.
        for task in range(self.num_tasks):
            if not self.predecessors[task]:
                self.state.ready[task] = True
                self.state.ready_time_sec[task] = self.release_time_sec[task]
        self.episode_end_sec = None if episode_end_sec is None else float(episode_end_sec)
        if self.episode_end_sec is not None and self.episode_end_sec <= float(start_sec):
            raise ValueError('Episode ends before DAG release')
        self.deadline_sec = np.asarray([p['deadline_sec'] for p in self.profiles], dtype=np.float64)
        self.trace = trace
        self.start_sec = float(start_sec)
        self.num_vehicles = NUM_VEHICLES
        self.num_edge_servers = NUM_EDGE_SERVERS
        self.num_resources = NUM_VEHICLES + NUM_EDGE_SERVERS
        legacy_vehicle_count = 6
        extra_vehicle_count = NUM_VEHICLES - legacy_vehicle_count
        extra_edge_count = NUM_EDGE_SERVERS - 1
        if extra_vehicle_count < 0 or extra_edge_count < 0:
            raise ValueError('Mobility mode must retain the original resources')
        for name, v_total, e_total in (
            ('total_a7', VEHICLE_A7_CORES, EDGE_A7_CORES),
            ('total_a12', VEHICLE_A12_CORES, EDGE_A12_CORES),
        ):
            previous = getattr(self, name)
            expanded = np.concatenate((previous[:6], np.full(extra_vehicle_count, v_total),
                                       previous[6:7], np.full(extra_edge_count, e_total)))
            setattr(self, name, expanded)
            setattr(self.state, name, expanded.copy())
        for core, upper_v, upper_e in (('a7', VEHICLE_A7_CORES, EDGE_A7_CORES),
                                        ('a12', VEHICLE_A12_CORES, EDGE_A12_CORES)):
            v = self.rng.integers(V4_REACHABLE_V2V_MIN_FREE_CORES, upper_v + 1,
                                  size=extra_vehicle_count, dtype=np.int64)
            e = self.rng.integers(V4_EDGE_MIN_FREE_CORES, upper_e + 1,
                                  size=extra_edge_count, dtype=np.int64)
            for prefix in ('scenario_capacity_', 'free_'):
                key = prefix + core
                prev = getattr(self.state, key)
                setattr(self.state, key, np.concatenate((prev[:6], v, prev[6:7], e)))
        assert len(self.state.free_a7) == len(self.state.free_a12) == self.num_resources
        if capacity_override is not None:
            for name in ('a7','a12'):
                capacity = np.asarray(capacity_override[name], dtype=np.int64).copy()
                total = self.total_a7 if name == 'a7' else self.total_a12
                if capacity.shape != total.shape or np.any(capacity < 0) or np.any(capacity > total):
                    raise ValueError('Invalid inherited resource capacity')
                setattr(self.state, 'scenario_capacity_'+name,capacity.copy())
                setattr(self.state, 'free_'+name,capacity.copy())

        n = self.num_tasks
        self.work_done = np.zeros(n, dtype=np.float64)  # fraction of original gang work
        self.energy_spent_j = np.zeros(n, dtype=np.float64)
        self.segment_begin_sec = np.full(n, -1., dtype=np.float64)
        self.event_sequence_for_task = np.full(n, -1, dtype=np.int64)
        self.awaiting_delivery = np.zeros(n, dtype=bool)
        self.delivery_time_sec = np.full(n, -1., dtype=np.float64)
        self.migration_count = np.zeros(n, dtype=np.int64)
        self.last_migration_time = np.full(n, -math.inf, dtype=np.float64)
        self.next_checkpoint_sec = np.full(n, math.inf, dtype=np.float64)
        self.migration_pending = set()
        self.migration_history = []
        self.delivery_history = []
        self.link_event_index = 0
        self.state.last_link_events = []
        if self.start_sec < self.trace.times[0] or self.start_sec >= self.trace.end:
            raise ValueError('Workflow start must be inside recorded SUMO trace')
        while (self.link_event_index < len(trace.events)
               and trace.events[self.link_event_index][0] <= self.start_sec + EPS):
            self.link_event_index += 1
        self._update_mobility()

    def _update_mobility(self):
        t = self.start_sec + float(self.state.simulation_time_sec)
        frame = self.trace.at(t)
        self.state.v2v_connected = set(frame['v2v_candidates'])
        self.state.edge_connected = set(frame['edge_candidates'])
        self.state.mobility = frame
        self.state._mobility_servers = self.trace.servers
        self.state.physical_time_sec = t

    def reachable(self, slot):
        slot = int(slot)
        if slot == 0:
            return True
        if 1 <= slot < NUM_VEHICLES:
            return slot in self.state.v2v_connected
        if NUM_VEHICLES <= slot < self.num_resources:
            return (slot - NUM_VEHICLES) in self.state.edge_connected
        return False

    def _resource_relation_exists(self, task_id, resource_slot):
        width = int(self.gang_size[task_id])
        slot = int(resource_slot)
        return self.reachable(slot) and (self.total_a7[slot] >= width or self.total_a12[slot] >= width)

    def _task_duration(self, task_id, processor):
        return float(self.profiles[int(task_id)]['a7_time_sec' if int(processor) == 0 else 'a12_time_sec'])

    def _task_power(self, task_id, processor):
        return float(self.profiles[int(task_id)]['a7_power_w' if int(processor) == 0 else 'a12_power_w'])

    def _reserve(self, slot, proc, width):
        pool = self.state.free_a7 if proc == 0 else self.state.free_a12
        if not (0 <= slot < self.num_resources) or pool[slot] < width:
            raise RuntimeError('Insufficient capacity to reserve task gang')
        pool[slot] -= width

    def _release(self, slot, proc, width):
        pool = self.state.free_a7 if proc == 0 else self.state.free_a12
        cap = self.state.scenario_capacity_a7 if proc == 0 else self.state.scenario_capacity_a12
        pool[slot] += width
        if pool[slot] > cap[slot]:
            raise RuntimeError('Over-released resource capacity')

    def _queue_completion(self, task_id, remaining_duration):
        self._event_sequence += 1
        seq = self._event_sequence
        self.event_sequence_for_task[task_id] = seq
        slot = int(self.state.assigned_resource_slot[task_id])
        proc = int(self.state.assigned_processor_index[task_id])
        width = int(self.state.assigned_cores[task_id])
        finish = float(self.state.simulation_time_sec) + remaining_duration
        self.state.task_finish_time_sec[task_id] = finish
        heapq.heappush(self._events, V4CompletionEvent(finish, seq, task_id, slot, proc, width))
        self.segment_begin_sec[task_id] = float(self.state.simulation_time_sec)
        self.next_checkpoint_sec[task_id] = (float(self.state.simulation_time_sec) +
             max(MIN_CHECK_INTERVAL, remaining_duration * MIGRATION_CHECK_FRACTION)) if remaining_duration > 2*MIN_CHECK_INTERVAL else math.inf

    def _start_task(self, action: V4ReferenceAction):
        # Keep v4 initial task state and core reservations unchanged.
        super()._start_task(action)
        t = int(action.task_id)
        self.event_sequence_for_task[t] = self._event_sequence
        self.segment_begin_sec[t] = float(self.state.simulation_time_sec)
        self.next_checkpoint_sec[t] = (float(self.state.simulation_time_sec) +
            max(MIN_CHECK_INTERVAL, float(action.duration_sec) * MIGRATION_CHECK_FRACTION)) if action.duration_sec > 2*MIN_CHECK_INTERVAL else math.inf

    def _settle_progress(self, task):
        if not self.state.running[task]:
            return
        elapsed = max(0., float(self.state.simulation_time_sec) - self.segment_begin_sec[task])
        processor = int(self.state.assigned_processor_index[task])
        self.work_done[task] = min(1., self.work_done[task] + elapsed / self._task_duration(task, processor))
        self.energy_spent_j[task] += elapsed * self._task_power(task, processor)
        self.segment_begin_sec[task] = float(self.state.simulation_time_sec)

    def valid_migrations(self):
        """Only running tasks at an event/checkpoint: KEEP + reachable migrations.

        All checkpoint bytes are transferred atomically at the decision instant;
        both current and target resource must be directly connected to V0.
        """
        result = []
        for task in sorted(self.migration_pending):
            if not self.state.running[task]:
                continue
            old = int(self.state.assigned_resource_slot[task])
            proc = int(self.state.assigned_processor_index[task])
            elapsed = max(0., self.state.simulation_time_sec - self.segment_begin_sec[task])
            progress = min(1., self.work_done[task] + elapsed/self._task_duration(task, proc))
            if progress >= 1. - EPS:
                continue
            remaining = max(0., 1. - progress)
            # KEEP requires no connection; CPU continues even after disconnection.
            result.append(MigrationChoice(task, old, proc, True, progress,
                                          remaining*self._task_duration(task, proc),
                                          remaining*self._task_duration(task, proc)*self._task_power(task, proc)))
            if not self.reachable(old) or self.migration_count[task] >= MAX_MIGRATIONS_PER_TASK:
                continue
            width = int(self.gang_size[task])
            for dest in range(self.num_resources):
                if dest == old or not self.reachable(dest):
                    continue
                for new_proc in (0, 1):
                    pool = self.state.free_a7 if new_proc == 0 else self.state.free_a12
                    total = self.total_a7 if new_proc == 0 else self.total_a12
                    if total[dest] < width or pool[dest] < width:
                        continue
                    duration = remaining*self._task_duration(task, new_proc)
                    result.append(MigrationChoice(task, dest, new_proc, False, progress,
                                                  duration, duration*self._task_power(task, new_proc)))
        return result

    def apply_migration(self, choice: MigrationChoice):
        legal = {(m.task_id,m.resource_slot,m.processor_index,m.is_keep):m for m in self.valid_migrations()}
        key = (choice.task_id,choice.resource_slot,choice.processor_index,choice.is_keep)
        if key not in legal:
            raise ValueError('Migration action no longer legal')
        choice = legal[key]
        task = int(choice.task_id)
        self.migration_pending.discard(task)
        if choice.is_keep:
            self._schedule_checkpoint(task)
            return {'event':'MIGRATION_KEEP','task_id':task}
        old = int(self.state.assigned_resource_slot[task])
        old_proc = int(self.state.assigned_processor_index[task])
        width = int(self.state.assigned_cores[task])
        self._settle_progress(task)
        if not (self.reachable(old) and self.reachable(choice.resource_slot)):
            raise RuntimeError('Migration requires source and target connectivity')
        self._reserve(choice.resource_slot,choice.processor_index,width)
        self._release(old,old_proc,width)
        self.state.assigned_resource_slot[task] = choice.resource_slot
        self.state.assigned_processor_index[task] = choice.processor_index
        self.migration_count[task] += 1
        self.last_migration_time[task] = self.state.simulation_time_sec
        # Old completion is stale: queue sequence becomes the sole valid one.
        self._queue_completion(task,(1.-self.work_done[task])*self._task_duration(task,choice.processor_index))
        event = {'event':'MIGRATED','task_id':task,'from_resource':old,
                 'to_resource':choice.resource_slot,'from_processor':old_proc,
                 'to_processor':choice.processor_index,'progress_saved':float(self.work_done[task]),
                 'time_sec':float(self.state.simulation_time_sec),'transfer_time_sec':0.,
                 'transfer_energy_j':0.}
        self.migration_history.append(event)
        return event

    def _schedule_checkpoint(self, task):
        if self.state.running[task]:
            remaining = max(0.,float(self.state.task_finish_time_sec[task]-self.state.simulation_time_sec))
            self.next_checkpoint_sec[task] = (self.state.simulation_time_sec +
                max(MIN_CHECK_INTERVAL, remaining*MIGRATION_CHECK_FRACTION)) if remaining > 2*MIN_CHECK_INTERVAL else math.inf
        else:
            self.next_checkpoint_sec[task] = math.inf

    def _peek_completion_time(self):
        while self._events and (not self.state.running[self._events[0].task_id] or
                 self.event_sequence_for_task[self._events[0].task_id] != self._events[0].sequence):
            heapq.heappop(self._events)
        return self._events[0].finish_time_sec if self._events else math.inf

    def _future_link_time(self):
        now = self.start_sec + float(self.state.simulation_time_sec)
        while self.link_event_index < len(self.trace.events) and self.trace.events[self.link_event_index][0] <= now+EPS:
            self.link_event_index += 1
        return self.trace.events[self.link_event_index][0]-self.start_sec if self.link_event_index < len(self.trace.events) else math.inf

    def _future_checkpoint_time(self):
        candidates = [self.next_checkpoint_sec[t] for t in range(self.num_tasks)
                      if self.state.running[t] and t not in self.migration_pending
                      and self.next_checkpoint_sec[t] > self.state.simulation_time_sec+EPS]
        return min(candidates,default=math.inf)

    def _deliver_available_results(self):
        delivered = []
        for task in np.flatnonzero(self.awaiting_delivery):
            if self.reachable(int(self.state.assigned_resource_slot[task])):
                self.awaiting_delivery[task] = False
                self.state.completed[task] = True
                self.delivery_time_sec[task] = float(self.state.simulation_time_sec)
                delivered.append(int(task))
                self.delivery_history.append({'task_id':int(task),'delivered_at_sec':float(self.state.simulation_time_sec),'buffered':True})
        return delivered

    def _process_next_completion_batch(self):
        now = float(self.state.simulation_time_sec)
        completed = []
        while self._events and self._events[0].finish_time_sec <= now + V4_EVENT_TIME_EPSILON_SEC:
            event = heapq.heappop(self._events)
            task = int(event.task_id)
            if not self.state.running[task] or self.event_sequence_for_task[task] != event.sequence:
                continue
            self._settle_progress(task)
            self.work_done[task] = 1.
            self._release(event.resource_slot,event.processor_index,event.cores)
            self.state.running[task] = False
            self.next_checkpoint_sec[task] = math.inf
            self.migration_pending.discard(task)
            if self.reachable(event.resource_slot):
                self.state.completed[task] = True
                self.delivery_time_sec[task] = now
                completed.append(task)
                self.delivery_history.append({'task_id':task,'delivered_at_sec':now,'buffered':False})
            else:
                self.awaiting_delivery[task] = True
        return completed

    def _mark_newly_ready_tasks(self):
        ready = []
        now = float(self.state.simulation_time_sec)
        for task in range(self.num_tasks):
            if self.state.ready[task] or self.state.running[task] or self.state.completed[task] or self.awaiting_delivery[task]:
                continue
            if self.state.task_start_time_sec[task] >= 0.:
                continue
            preds = self.predecessors[task]
            if preds and all(self.state.completed[p] for p in preds):
                available = max(float(self.state.release_time_sec[task]),max(self.delivery_time_sec[p] for p in preds))
                if available <= now + EPS:
                    self.state.ready[task] = True
                    self.state.ready_time_sec[task] = available
                    ready.append(task)
        return tuple(ready)

    def advance_one_event(self):
        """Causal event integration: completed work, link, checkpoint, delivery."""
        comp = self._peek_completion_time()
        link = self._future_link_time()
        checkpoint = self._future_checkpoint_time()
        t = min(comp,link,checkpoint)
        deadline = math.inf if self.episode_end_sec is None else self.episode_end_sec - self.start_sec
        if deadline < t - EPS:
            # Settle *partial* work for accurate energy usage at t=600;
            # do not fabricate completion or delivery at the cutoff.
            self.state.simulation_time_sec = float(deadline)
            self._update_mobility()
            for task in np.flatnonzero(self.state.running):
                self._settle_progress(int(task))
            raise WorkflowTimeLimitReached('600-second SUMO episode boundary; DAG incomplete')
        if not math.isfinite(t):
            raise RuntimeError('No future event; pending tasks or buffered results cannot finish in this trace')
        if self.start_sec+t > self.trace.end+EPS:
            raise RuntimeError('SUMO trace too short to complete workflow')
        self.state.simulation_time_sec = float(t)
        self._update_mobility()  # use contact state AT event time; no future information
        completed = self._process_next_completion_batch() if comp <= t+EPS else []
        changes = []
        if link <= t+EPS:
            while (self.link_event_index < len(self.trace.events)
                   and abs(self.trace.events[self.link_event_index][0]-(self.start_sec+t)) <= 1e-7):
                changes.extend(self.trace.events[self.link_event_index][1])
                self.link_event_index += 1
        delivered = self._deliver_available_results()
        completed.extend(delivered)
        newly_ready = self._mark_newly_ready_tasks()
        if link <= t+EPS or checkpoint <= t+EPS or completed:
            self.migration_pending.update(int(i) for i in np.flatnonzero(self.state.running))
        for i in self.migration_pending:
            self.next_checkpoint_sec[i] = math.inf
        self.state.last_link_events = changes
        return {'completed_tasks':list(sorted(completed)), 'newly_ready_tasks':list(newly_ready),
                'link_events':changes, 'trigger_event':
                'task_completion' if completed else 'link_change' if changes else 'migration_checkpoint'}

    def iter_decision_epochs(self):
        roots = tuple(int(x) for x in np.flatnonzero(self.state.ready))
        yield self._make_epoch('initial_ready_tasks', affected_tasks=roots, newly_ready_tasks=roots)
        while not bool(np.all(self.state.completed)):
            if self._decision_epoch >= 100000:
                raise RuntimeError('Decision event limit exceeded')
            action = self._select_action()
            if action is not None:
                self.state.last_link_events=[]
                self._start_task(action)
                yield self._make_epoch('task_started', affected_tasks=(action.task_id,), started_task=action.task_id)
            else:
                event = self.advance_one_event()
                affected = tuple(sorted(set(event['completed_tasks']+event['newly_ready_tasks'])))
                yield self._make_epoch(event['trigger_event'], affected_tasks=affected,
                        completed_tasks=event['completed_tasks'], newly_ready_tasks=event['newly_ready_tasks'])
