from __future__ import annotations

import random

from warehouse_simulator.strategies.base import StorageLocationStrategy
from warehouse_simulator.utils import chunk_evenly


class RandomStrategy(StorageLocationStrategy):
    name = "Random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def assign(
        self,
        sku_list: list[int],
        warehouse_map,
        order_data: list[dict],
    ) -> dict[int, list[int]]:
        rng = random.Random(self.seed)
        shuffled = list(sku_list)
        rng.shuffle(shuffled)
        buckets = chunk_evenly(shuffled, warehouse_map.num_lanes)
        return {lane_id: buckets[lane_id] for lane_id in range(warehouse_map.num_lanes)}

