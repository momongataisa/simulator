from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from warehouse_simulator.config import WarehouseConfig
from warehouse_simulator.utils import Position


@dataclass(frozen=True)
class Cell:
    position: Position
    kind: str


class WarehouseMap:
    def __init__(self, config: WarehouseConfig):
        self.config = config
        self.height = config.grid_height
        self.width = config.grid_width
        self.num_lanes = config.num_lanes
        self.num_export_ports = config.num_export_ports
        self.cross_aisle_y = 0

        self.lane_columns: list[int] = []
        self.lane_cells: dict[int, list[Position]] = {}
        self.lane_storage_cells: dict[int, list[Position]] = {}
        self.coord_to_lane: dict[Position, int] = {}
        self.export_ports: dict[int, Position] = {}
        self.buffers: dict[int, list[Position]] = {}

        self.traversable: set[Position] = set()
        self._adjacency: dict[Position, list[Position]] = {}

        self.lane_to_skus: dict[int, list[int]] = defaultdict(list)
        self.sku_to_coord: dict[int, Position] = {}

        self._build_layout()

    def _build_layout(self) -> None:
        self.lane_columns = self._choose_columns(self.num_lanes)
        export_columns = self._choose_columns(self.num_export_ports, use_lane_columns=True)

        for x in range(self.width):
            self.traversable.add((self.cross_aisle_y, x))

        for lane_id, x in enumerate(self.lane_columns):
            lane_positions = [(y, x) for y in range(1, self.height)]
            storage_positions = [(y, x) for y in range(2, self.height)]
            self.lane_cells[lane_id] = lane_positions
            self.lane_storage_cells[lane_id] = storage_positions
            for pos in lane_positions:
                self.traversable.add(pos)
                self.coord_to_lane[pos] = lane_id

        for port_id, x in enumerate(export_columns):
            port_pos = (self.cross_aisle_y, x)
            self.export_ports[port_id] = port_pos
            neighbor_cells: list[Position] = []
            for dx in (-1, 1):
                bx = x + dx
                if 0 <= bx < self.width:
                    neighbor_cells.append((self.cross_aisle_y, bx))
            self.buffers[port_id] = neighbor_cells

        self._adjacency = {}
        for node in self.traversable:
            neighbors: list[Position] = []
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (node[0] + dy, node[1] + dx)
                if nxt in self.traversable:
                    neighbors.append(nxt)
            self._adjacency[node] = neighbors

    def _choose_columns(self, count: int, use_lane_columns: bool = False) -> list[int]:
        if count <= 0:
            return []

        if use_lane_columns and self.lane_columns:
            source = self.lane_columns
        else:
            source = list(range(self.width))

        if count >= len(source):
            return list(source[:count])

        if count == 1:
            return [source[len(source) // 2]]

        step = (len(source) - 1) / (count - 1)
        indices = [int(round(i * step)) for i in range(count)]
        selected = [source[index] for index in indices]
        return selected

    def neighbors(self, pos: Position) -> list[Position]:
        return self._adjacency.get(pos, [])

    def is_traversable(self, pos: Position) -> bool:
        return pos in self.traversable

    def lane_for_position(self, pos: Position) -> int | None:
        return self.coord_to_lane.get(pos)

    def get_graph(self) -> tuple[list[Position], list[tuple[Position, Position]]]:
        nodes = list(self.traversable)
        edges: list[tuple[Position, Position]] = []
        for node, neighbors in self._adjacency.items():
            for neighbor in neighbors:
                if node < neighbor:
                    edges.append((node, neighbor))
        return nodes, edges

    def get_export_port_position(self, export_port_id: int) -> Position:
        if export_port_id not in self.export_ports:
            export_port_id %= max(1, len(self.export_ports))
        return self.export_ports[export_port_id]

    def assign_skus_to_lanes(self, lane_to_skus: dict[int, list[int]]) -> None:
        self.lane_to_skus = defaultdict(list)
        self.sku_to_coord = {}

        seen: set[int] = set()
        for lane_id in range(self.num_lanes):
            skus = list(lane_to_skus.get(lane_id, []))
            storage_slots = self.lane_storage_cells[lane_id]
            if len(skus) > len(storage_slots):
                raise ValueError(
                    f"Lane {lane_id} has {len(skus)} SKUs but only {len(storage_slots)} slots."
                )
            for idx, sku in enumerate(skus):
                if sku in seen:
                    raise ValueError(f"Duplicate SKU allocation detected: {sku}")
                seen.add(sku)
                self.lane_to_skus[lane_id].append(sku)
                self.sku_to_coord[sku] = storage_slots[idx]

    def get_sku_position(self, sku_id: int) -> Position:
        if sku_id not in self.sku_to_coord:
            raise KeyError(f"SKU {sku_id} is not placed in the warehouse map.")
        return self.sku_to_coord[sku_id]

