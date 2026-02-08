from __future__ import annotations

import heapq
from collections import defaultdict

from warehouse_simulator.utils import Position, manhattan


class PathPlanner:
    def __init__(
        self,
        warehouse_map,
        planning_horizon: int = 120,
        pickup_wait_steps: int = 1,
    ):
        self.warehouse_map = warehouse_map
        self.planning_horizon = planning_horizon
        self.pickup_wait_steps = max(0, pickup_wait_steps)

    def plan_for_agents(
        self,
        agents: dict[int, "Agent"],
        tasks: dict[int, "Task"],
        now_step: int,
        agent_ids: list[int],
    ) -> None:
        if not agent_ids:
            return

        replanning = set(agent_ids)
        reservations = self._build_reservations(agents, now_step, ignore_agent_ids=replanning)

        for agent_id in sorted(replanning):
            agent = agents[agent_id]
            if agent.assigned_task_id is None:
                continue
            task = tasks[agent.assigned_task_id]
            full_path = self._plan_task_path(
                agent_id=agent_id,
                start=agent.position,
                pickup=task.start,
                goal=task.goal,
                now_step=now_step,
                reservations=reservations,
            )
            if full_path is None:
                full_path = [agent.position]
            agent.set_path(full_path)
            self._reserve_path(
                reservations,
                full_path,
                start_time=now_step,
                agent_id=agent_id,
            )

    def _plan_task_path(
        self,
        agent_id: int,
        start: Position,
        pickup: Position,
        goal: Position,
        now_step: int,
        reservations,
    ) -> list[Position] | None:
        path_to_pickup = self._astar_time_expanded(
            start=start,
            goal=pickup,
            start_time=now_step,
            agent_id=agent_id,
            reservations=reservations,
        )
        if not path_to_pickup:
            return None

        pickup_wait = [pickup] * self.pickup_wait_steps
        second_leg_start_time = now_step + len(path_to_pickup) - 1 + self.pickup_wait_steps
        path_to_goal = self._astar_time_expanded(
            start=pickup,
            goal=goal,
            start_time=second_leg_start_time,
            agent_id=agent_id,
            reservations=reservations,
        )
        if not path_to_goal:
            return None

        return path_to_pickup + pickup_wait + path_to_goal[1:]

    def _astar_time_expanded(
        self,
        start: Position,
        goal: Position,
        start_time: int,
        agent_id: int,
        reservations,
    ) -> list[Position] | None:
        if not self.warehouse_map.is_traversable(start) or not self.warehouse_map.is_traversable(goal):
            return None

        open_heap: list[tuple[int, int, Position, int]] = []
        came_from: dict[tuple[Position, int], tuple[Position, int] | None] = {}
        cost_so_far: dict[tuple[Position, int], int] = {}

        start_state = (start, start_time)
        came_from[start_state] = None
        cost_so_far[start_state] = 0
        heapq.heappush(open_heap, (manhattan(start, goal), 0, start, start_time))

        while open_heap:
            _priority, path_cost, current_pos, current_time = heapq.heappop(open_heap)
            if current_pos == goal:
                return self._reconstruct_path(came_from, (current_pos, current_time))

            if current_time - start_time >= self.planning_horizon:
                continue

            candidate_neighbors = [current_pos] + self.warehouse_map.neighbors(current_pos)
            for next_pos in candidate_neighbors:
                next_time = current_time + 1
                if next_time - start_time > self.planning_horizon:
                    continue
                if self._violates_constraints(
                    current_pos=current_pos,
                    next_pos=next_pos,
                    current_time=current_time,
                    next_time=next_time,
                    agent_id=agent_id,
                    reservations=reservations,
                ):
                    continue

                next_state = (next_pos, next_time)
                next_cost = path_cost + 1
                if next_cost < cost_so_far.get(next_state, 10**9):
                    cost_so_far[next_state] = next_cost
                    came_from[next_state] = (current_pos, current_time)
                    priority = next_cost + manhattan(next_pos, goal)
                    heapq.heappush(open_heap, (priority, next_cost, next_pos, next_time))

        return None

    def _reconstruct_path(
        self,
        came_from: dict[tuple[Position, int], tuple[Position, int] | None],
        goal_state: tuple[Position, int],
    ) -> list[Position]:
        path: list[Position] = []
        state: tuple[Position, int] | None = goal_state
        while state is not None:
            path.append(state[0])
            state = came_from[state]
        path.reverse()
        return path

    def _violates_constraints(
        self,
        current_pos: Position,
        next_pos: Position,
        current_time: int,
        next_time: int,
        agent_id: int,
        reservations,
    ) -> bool:
        if not self.warehouse_map.is_traversable(next_pos):
            return True

        reserved_vertex = reservations["vertex"].get(next_time, {})
        if next_pos in reserved_vertex and reserved_vertex[next_pos] != agent_id:
            return True

        reserved_edges = reservations["edge"].get(current_time, set())
        if (next_pos, current_pos) in reserved_edges:
            return True

        lane_id = self.warehouse_map.lane_for_position(next_pos)
        if lane_id is not None:
            reserved_lane = reservations["lane"].get(next_time, {})
            if lane_id in reserved_lane and reserved_lane[lane_id] != agent_id:
                return True

        return False

    def _build_reservations(
        self,
        agents: dict[int, "Agent"],
        now_step: int,
        ignore_agent_ids: set[int],
    ):
        reservations = {
            "vertex": defaultdict(dict),
            "edge": defaultdict(set),
            "lane": defaultdict(dict),
        }
        for agent in agents.values():
            if agent.agent_id in ignore_agent_ids:
                continue

            remaining_path = agent.path[agent.path_index :] if agent.path else [agent.position]
            if not remaining_path:
                remaining_path = [agent.position]
            self._reserve_path(
                reservations,
                remaining_path,
                start_time=now_step,
                agent_id=agent.agent_id,
            )
        return reservations

    def _reserve_path(self, reservations, path: list[Position], start_time: int, agent_id: int) -> None:
        if not path:
            return
        for offset, pos in enumerate(path):
            time_step = start_time + offset
            reservations["vertex"][time_step][pos] = agent_id
            lane_id = self.warehouse_map.lane_for_position(pos)
            if lane_id is not None:
                reservations["lane"][time_step][lane_id] = agent_id
            if offset > 0:
                prev = path[offset - 1]
                reservations["edge"][time_step - 1].add((prev, pos))

        final_pos = path[-1]
        final_lane = self.warehouse_map.lane_for_position(final_pos)
        for extra_time in range(start_time + len(path), start_time + self.planning_horizon + 1):
            reservations["vertex"][extra_time][final_pos] = agent_id
            if final_lane is not None:
                reservations["lane"][extra_time][final_lane] = agent_id

