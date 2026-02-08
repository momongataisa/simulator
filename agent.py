from __future__ import annotations

from dataclasses import dataclass, field

from warehouse_simulator.utils import Position


@dataclass
class Agent:
    agent_id: int
    position: Position
    assigned_task_id: int | None = None
    carrying_sku: bool = False
    path: list[Position] = field(default_factory=list)
    path_index: int = 0

    def __post_init__(self) -> None:
        if not self.path:
            self.path = [self.position]

    @property
    def is_free(self) -> bool:
        return self.assigned_task_id is None

    def assign_task(self, task_id: int) -> None:
        self.assigned_task_id = task_id
        self.carrying_sku = False
        self.path = [self.position]
        self.path_index = 0

    def clear_task(self) -> None:
        self.assigned_task_id = None
        self.carrying_sku = False
        self.path = [self.position]
        self.path_index = 0

    def set_path(self, path: list[Position]) -> None:
        if not path:
            self.path = [self.position]
            self.path_index = 0
            return
        self.path = path
        self.path_index = 0

    def has_next_step(self) -> bool:
        return self.path_index < len(self.path) - 1

    def advance_one_step(self) -> None:
        if self.has_next_step():
            self.path_index += 1
            self.position = self.path[self.path_index]

