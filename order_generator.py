from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from warehouse_simulator.config import OrderConfig


@dataclass
class GeneratedOrders:
    train_orders: list[dict]
    test_orders: list[dict]
    co_occurrence_groups: list[list[int]]
    sku_probabilities: np.ndarray


class OrderGenerator:
    def __init__(self, config: OrderConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate(self, num_export_ports: int) -> GeneratedOrders:
        sku_ids = np.arange(self.config.num_skus)
        sku_probabilities = self._build_sku_probability()
        co_groups = self._build_co_groups()

        orders: list[dict] = []
        for order_id in range(self.config.num_orders):
            order_size = int(
                self.rng.integers(
                    self.config.min_items_per_order,
                    self.config.max_items_per_order + 1,
                )
            )
            sku_list = self._sample_order_items(sku_ids, sku_probabilities, co_groups, order_size)
            export_port_id = int(self.rng.integers(0, max(1, num_export_ports)))
            orders.append(
                {
                    "order_id": order_id,
                    "sku_list": sku_list,
                    "export_port_id": export_port_id,
                }
            )

        permutation = self.rng.permutation(len(orders))
        split_index = int(len(orders) * self.config.train_ratio)
        train_indices = permutation[:split_index]
        test_indices = permutation[split_index:]

        train_orders = [orders[i] for i in train_indices]
        test_orders = [orders[i] for i in test_indices]
        return GeneratedOrders(train_orders, test_orders, co_groups, sku_probabilities)

    def _build_sku_probability(self) -> np.ndarray:
        ranks = np.arange(1, self.config.num_skus + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, self.config.popularity_skew)
        weights /= np.sum(weights)
        return weights

    def _build_co_groups(self) -> list[list[int]]:
        groups: list[list[int]] = []
        min_group_size = max(3, self.config.max_items_per_order)
        max_group_size = max(min_group_size + 1, min(20, self.config.num_skus // 3))

        for _ in range(self.config.topic_count):
            size = int(self.rng.integers(min_group_size, max_group_size + 1))
            group = self.rng.choice(self.config.num_skus, size=size, replace=False).tolist()
            groups.append(group)
        return groups

    def _sample_order_items(
        self,
        sku_ids: np.ndarray,
        sku_probabilities: np.ndarray,
        co_groups: list[list[int]],
        order_size: int,
    ) -> list[int]:
        order_size = min(order_size, len(sku_ids))
        use_group = bool(co_groups) and self.rng.random() < self.config.co_occurrence_strength

        if use_group:
            group = co_groups[int(self.rng.integers(0, len(co_groups)))]
            pool = np.array(group, dtype=np.int64)
            if len(pool) >= order_size:
                probs = sku_probabilities[pool]
                probs /= np.sum(probs)
                selected = self.rng.choice(pool, size=order_size, replace=False, p=probs)
                return selected.tolist()

        selected = self.rng.choice(
            sku_ids,
            size=order_size,
            replace=False,
            p=sku_probabilities,
        )
        return selected.tolist()


def build_sku_frequency(order_data: list[dict], num_skus: int) -> np.ndarray:
    frequencies = np.zeros(num_skus, dtype=np.float64)
    for order in order_data:
        for sku_id in order["sku_list"]:
            if 0 <= sku_id < num_skus:
                frequencies[sku_id] += 1

    total = float(np.sum(frequencies))
    if total <= 0:
        return np.ones(num_skus, dtype=np.float64) / max(1, num_skus)
    return frequencies / total

