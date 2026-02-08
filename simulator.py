from __future__ import annotations

from dataclasses import dataclass

from warehouse_simulator.agent import Agent
from warehouse_simulator.config import OrderConfig, SimulationConfig, WarehouseConfig
from warehouse_simulator.map import WarehouseMap
from warehouse_simulator.order_generator import OrderGenerator
from warehouse_simulator.path_planner import PathPlanner
from warehouse_simulator.task_manager import TaskManager
from warehouse_simulator.utils import set_global_seed


@dataclass
class SimulationResult:
    method: str
    throughput: float
    completed_tasks: int
    total_tasks: int
    elapsed_steps: int
    pending_tasks: int
    per_step_throughput: list[float]


class Simulator:
    def __init__(
        self,
        warehouse_config: WarehouseConfig,
        order_config: OrderConfig,
        simulation_config: SimulationConfig,
    ):
        self.warehouse_config = warehouse_config
        self.order_config = order_config
        self.simulation_config = simulation_config

    def run(self, strategy, seed: int | None = None) -> SimulationResult:
        active_seed = self.simulation_config.seed if seed is None else seed
        set_global_seed(active_seed)

        warehouse_map = WarehouseMap(self.warehouse_config)
        generated_orders = OrderGenerator(self.order_config, seed=active_seed).generate(
            warehouse_map.num_export_ports
        )

        sku_list = list(range(self.order_config.num_skus))
        lane_allocation = strategy.assign(sku_list, warehouse_map, generated_orders.train_orders)
        warehouse_map.assign_skus_to_lanes(lane_allocation)

        agents = self._initialize_agents(warehouse_map)
        task_manager = TaskManager(list(warehouse_map.export_ports.keys()))
        task_manager.build_tasks(generated_orders.test_orders, warehouse_map)

        planner = PathPlanner(
            warehouse_map=warehouse_map,
            planning_horizon=self.simulation_config.plan_horizon,
            pickup_wait_steps=self.simulation_config.pickup_wait_steps,
        )

        per_step_throughput: list[float] = []
        elapsed_steps = 0

        for step in range(self.simulation_config.max_steps):
            elapsed_steps = step + 1

            task_manager.assign_tasks_greedily(agents)
            need_planning = self._agents_needing_path(agents)
            planner.plan_for_agents(agents, task_manager.tasks, step, need_planning)

            self._advance_agents(agents, task_manager)
            completed = task_manager.completed_count()
            per_step_throughput.append(completed / elapsed_steps)

            if task_manager.all_done():
                break

        completed_tasks = task_manager.completed_count()
        total_tasks = len(task_manager.tasks)
        throughput = completed_tasks / max(1, elapsed_steps)

        return SimulationResult(
            method=strategy.name,
            throughput=throughput,
            completed_tasks=completed_tasks,
            total_tasks=total_tasks,
            elapsed_steps=elapsed_steps,
            pending_tasks=task_manager.pending_count(),
            per_step_throughput=per_step_throughput,
        )

    def _initialize_agents(self, warehouse_map: WarehouseMap) -> dict[int, Agent]:
        export_positions = list(warehouse_map.export_ports.values())
        if not export_positions:
            export_positions = [(0, 0)]

        agents: dict[int, Agent] = {}
        for agent_id in range(self.warehouse_config.num_agents):
            start_position = export_positions[agent_id % len(export_positions)]
            agents[agent_id] = Agent(agent_id=agent_id, position=start_position)
        return agents

    def _agents_needing_path(self, agents: dict[int, Agent]) -> list[int]:
        need_path: list[int] = []
        for agent in agents.values():
            if agent.assigned_task_id is None:
                continue
            if not agent.has_next_step():
                need_path.append(agent.agent_id)
        return need_path

    def _advance_agents(self, agents: dict[int, Agent], task_manager: TaskManager) -> None:
        for agent in agents.values():
            agent.advance_one_step()
            if agent.assigned_task_id is None:
                continue

            task = task_manager.tasks[agent.assigned_task_id]
            if not agent.carrying_sku and agent.position == task.start:
                agent.carrying_sku = True

            if agent.carrying_sku and agent.position == task.goal:
                task_manager.complete_task(task.task_id)
                agent.clear_task()
