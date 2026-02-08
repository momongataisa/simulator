from __future__ import annotations

import numpy as np

from warehouse_simulator.order_generator import build_sku_frequency
from warehouse_simulator.strategies.base import StorageLocationStrategy
from warehouse_simulator.utils import target_bucket_sizes


class FrequencyStrategy(StorageLocationStrategy):
    name = "Frequency"

    def assign(
        self,
        sku_list: list[int],
        warehouse_map,
        order_data: list[dict],
    ) -> dict[int, list[int]]:
        num_lanes = warehouse_map.num_lanes
        target_sizes = target_bucket_sizes(len(sku_list), num_lanes)
        sku_frequency = build_sku_frequency(order_data, len(sku_list))

        allocation: dict[int, list[int]] = {lane: [] for lane in range(num_lanes)}
        lane_loads = np.zeros(num_lanes, dtype=np.float64)
        sorted_skus = sorted(sku_list, key=lambda sku: sku_frequency[sku], reverse=True)

        for sku_id in sorted_skus:
            candidate_lanes = [
                lane
                for lane in range(num_lanes)
                if len(allocation[lane]) < target_sizes[lane]
            ]
            if not candidate_lanes:
                candidate_lanes = list(range(num_lanes))
            best_lane = min(candidate_lanes, key=lambda lane: lane_loads[lane])
            allocation[best_lane].append(sku_id)
            lane_loads[best_lane] += sku_frequency[sku_id]

        return allocation

