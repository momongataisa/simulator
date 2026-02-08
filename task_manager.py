from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from warehouse_simulator.utils import Position, manhattan


@dataclass
class Task:
    task_id: int
    order_id: int
    sku_id: int
    start: Position
    goal: Position
    export_port_id: int
    status: str = "pending"
    assigned_agent_id: int | None = None


class TaskManager:
    def __init__(self, export_port_ids: list[int]):
        self.export_port_ids = export_port_ids
        self.tasks: dict[int, Task] = {}
        self.port_queues: dict[int, deque[int]] = {port_id: deque() for port_id in export_port_ids}
        self.pending_task_ids: set[int] = set()
        self.completed_task_ids: set[int] = set()

    def build_tasks(self, orders: list[dict], warehouse_map) -> None:
        task_id = 0
        for order in orders:
            export_port_id = int(order["export_port_id"])
            if export_port_id not in self.port_queues:
                export_port_id %= max(1, len(self.port_queues))
            goal = warehouse_map.get_export_port_position(export_port_id)
            for sku_id in order["sku_list"]:
                if sku_id not in warehouse_map.sku_to_coord:
                    continue
                start = warehouse_map.get_sku_position(sku_id)
                task = Task(
                    task_id=task_id,
                    order_id=int(order["order_id"]),
                    sku_id=int(sku_id),
                    start=start,
                    goal=goal,
                    export_port_id=export_port_id,
                )
                self.tasks[task_id] = task
                self.port_queues[export_port_id].append(task_id)
                self.pending_task_ids.add(task_id)
                task_id += 1

    def assign_tasks_greedily(self, agents: dict[int, "Agent"]) -> list[tuple[int, int]]:
        assignments: list[tuple[int, int]] = []
        if not self.pending_task_ids:
            return assignments

        for agent in sorted(agents.values(), key=lambda item: item.agent_id):
            if not agent.is_free:
                continue
            chosen_task_id = self._nearest_pending_task_id(agent.position)
            if chosen_task_id is None:
                continue
            task = self.tasks[chosen_task_id]
            task.status = "assigned"
            task.assigned_agent_id = agent.agent_id
            self.pending_task_ids.discard(chosen_task_id)
            agent.assign_task(chosen_task_id)
            assignments.append((agent.agent_id, chosen_task_id))
        return assignments

    def complete_task(self, task_id: int) -> None:
        task = self.tasks[task_id]
        task.status = "done"
        task.assigned_agent_id = None
        self.completed_task_ids.add(task_id)

    def all_done(self) -> bool:
        return len(self.completed_task_ids) == len(self.tasks)

    def pending_count(self) -> int:
        return len(self.pending_task_ids)

    def completed_count(self) -> int:
        return len(self.completed_task_ids)

    def _nearest_pending_task_id(self, current_position: Position) -> int | None:
        if not self.pending_task_ids:
            return None
        return min(
            self.pending_task_ids,
            key=lambda task_id: manhattan(current_position, self.tasks[task_id].start),
        )

