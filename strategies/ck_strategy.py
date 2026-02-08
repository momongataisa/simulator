from __future__ import annotations

from collections import defaultdict

import numpy as np

from warehouse_simulator.strategies.base import StorageLocationStrategy
from warehouse_simulator.utils import target_bucket_sizes


class CKStrategy(StorageLocationStrategy):
    name = "CK"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def assign(
        self,
        sku_list: list[int],
        warehouse_map,
        order_data: list[dict],
    ) -> dict[int, list[int]]:
        num_lanes = warehouse_map.num_lanes
        target_sizes = target_bucket_sizes(len(sku_list), num_lanes)
        features = self._build_co_occurrence_features(sku_list, order_data)
        labels, centers = self._fit_kmeans(features, num_lanes)

        allocation = {lane: [] for lane in range(num_lanes)}
        for sku_idx, lane in enumerate(labels):
            allocation[int(lane)].append(sku_idx)

        self._rebalance(allocation, target_sizes, features, centers)
        return allocation

    def _build_co_occurrence_features(
        self,
        sku_list: list[int],
        order_data: list[dict],
    ) -> np.ndarray:
        num_skus = len(sku_list)
        features = np.zeros((num_skus, num_skus), dtype=np.float64)

        for order in order_data:
            unique_skus = sorted(set(int(sku) for sku in order["sku_list"] if 0 <= int(sku) < num_skus))
            for sku in unique_skus:
                features[sku, sku] += 1.0
            for left_index in range(len(unique_skus)):
                left = unique_skus[left_index]
                for right in unique_skus[left_index + 1 :]:
                    features[left, right] += 1.0
                    features[right, left] += 1.0

        row_sums = np.sum(features, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return features / row_sums

    def _fit_kmeans(self, features: np.ndarray, num_lanes: int) -> tuple[np.ndarray, np.ndarray | None]:
        num_skus = features.shape[0]
        effective_clusters = max(1, min(num_lanes, num_skus))

        try:
            from sklearn.cluster import KMeans

            model = KMeans(
                n_clusters=effective_clusters,
                random_state=self.seed,
                n_init=10,
            )
            labels = model.fit_predict(features)
            centers = model.cluster_centers_
        except Exception:
            rng = np.random.default_rng(self.seed)
            labels = rng.integers(0, effective_clusters, size=num_skus)
            centers = None

        if effective_clusters < num_lanes:
            remap = np.array(labels, copy=True)
            labels = remap
        return labels, centers

    def _rebalance(
        self,
        allocation: dict[int, list[int]],
        target_sizes: list[int],
        features: np.ndarray,
        centers: np.ndarray | None,
    ) -> None:
        underfull: set[int] = set()
        for lane_id, target in enumerate(target_sizes):
            if len(allocation[lane_id]) < target:
                underfull.add(lane_id)

        for lane_id, target in enumerate(target_sizes):
            while len(allocation[lane_id]) > target:
                sku = allocation[lane_id].pop()
                if not underfull:
                    break
                destination = self._select_destination_lane(
                    sku=sku,
                    candidate_lanes=underfull,
                    features=features,
                    centers=centers,
                )
                allocation[destination].append(sku)
                if len(allocation[destination]) >= target_sizes[destination]:
                    underfull.discard(destination)

        missing = self._find_missing_skus(allocation, features.shape[0])
        if missing:
            lane_iter = 0
            for sku in missing:
                while target_sizes[lane_iter] <= len(allocation[lane_iter]):
                    lane_iter = (lane_iter + 1) % len(target_sizes)
                allocation[lane_iter].append(sku)
                lane_iter = (lane_iter + 1) % len(target_sizes)

    def _select_destination_lane(
        self,
        sku: int,
        candidate_lanes: set[int],
        features: np.ndarray,
        centers: np.ndarray | None,
    ) -> int:
        candidate_list = sorted(candidate_lanes)
        if centers is None or len(centers) == 0:
            return candidate_list[0]

        best_lane = candidate_list[0]
        best_distance = float("inf")
        for lane_id in candidate_list:
            if lane_id >= len(centers):
                return lane_id
            distance = float(np.linalg.norm(features[sku] - centers[lane_id]))
            if distance < best_distance:
                best_distance = distance
                best_lane = lane_id
        return best_lane

    def _find_missing_skus(self, allocation: dict[int, list[int]], num_skus: int) -> list[int]:
        assigned = set()
        duplicates = defaultdict(int)
        for lane_items in allocation.values():
            for sku in lane_items:
                duplicates[sku] += 1
                assigned.add(sku)

        for sku, count in duplicates.items():
            if count > 1:
                remove_count = count - 1
                for lane_items in allocation.values():
                    while remove_count > 0 and sku in lane_items:
                        lane_items.remove(sku)
                        remove_count -= 1

        missing = [sku for sku in range(num_skus) if sku not in assigned]
        return missing

