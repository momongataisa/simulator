from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict

import numpy as np

from warehouse_simulator.strategies.base import StorageLocationStrategy
from warehouse_simulator.strategies.ck_strategy import CKStrategy


class SLLDAStrategy(StorageLocationStrategy):
    name = "SL-LDA"

    def __init__(
        self,
        seed: int = 42,
        lda_topics: int = 8,
        lda_top_n: int = 12,
        lda_iterations: int = 150,
        sa_iterations: int = 1800,
        sa_initial_temp: float = 40.0,
        sa_cooling_rate: float = 0.996,
        loss_w1: float = 1.0,
        loss_w2: float = 1.8,
    ):
        self.seed = seed
        self.lda_topics = lda_topics
        self.lda_top_n = lda_top_n
        self.lda_iterations = lda_iterations
        self.sa_iterations = sa_iterations
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.loss_w1 = loss_w1
        self.loss_w2 = loss_w2

    def assign(
        self,
        sku_list: list[int],
        warehouse_map,
        order_data: list[dict],
    ) -> dict[int, list[int]]:
        rng = np.random.default_rng(self.seed)
        ck_allocation = CKStrategy(seed=self.seed).assign(sku_list, warehouse_map, order_data)
        co_sets = self._extract_co_sets(order_data, warehouse_map.num_lanes)
        if not co_sets:
            return ck_allocation

        current = copy.deepcopy(ck_allocation)
        current_loss = self._co_occurrence_loss(current, co_sets, warehouse_map.num_lanes)
        best = copy.deepcopy(current)
        best_loss = current_loss
        temperature = self.sa_initial_temp

        for _ in range(self.sa_iterations):
            candidate = self._generate_candidate(current, rng)
            candidate_loss = self._co_occurrence_loss(candidate, co_sets, warehouse_map.num_lanes)
            delta = candidate_loss - current_loss

            accept = delta <= 0
            if not accept:
                threshold = math.exp(-delta / max(1e-9, temperature))
                accept = rng.random() < threshold

            if accept:
                current = candidate
                current_loss = candidate_loss
                if current_loss < best_loss:
                    best = copy.deepcopy(current)
                    best_loss = current_loss

            temperature *= self.sa_cooling_rate

        return best

    def _generate_candidate(
        self,
        allocation: dict[int, list[int]],
        rng: np.random.Generator,
    ) -> dict[int, list[int]]:
        candidate = {lane: list(items) for lane, items in allocation.items()}
        lanes = sorted(candidate.keys())

        if len(lanes) < 2:
            return candidate

        if rng.random() < 0.25:
            lane_a, lane_b = rng.choice(lanes, size=2, replace=False).tolist()
            candidate[lane_a], candidate[lane_b] = candidate[lane_b], candidate[lane_a]
            return candidate

        non_empty_lanes = [lane for lane in lanes if candidate[lane]]
        if len(non_empty_lanes) < 2:
            return candidate

        lane_a, lane_b = rng.choice(non_empty_lanes, size=2, replace=False).tolist()
        idx_a = int(rng.integers(0, len(candidate[lane_a])))
        idx_b = int(rng.integers(0, len(candidate[lane_b])))
        candidate[lane_a][idx_a], candidate[lane_b][idx_b] = (
            candidate[lane_b][idx_b],
            candidate[lane_a][idx_a],
        )
        return candidate

    def _co_occurrence_loss(
        self,
        allocation: dict[int, list[int]],
        co_sets: list[set[int]],
        num_lanes: int,
    ) -> float:
        lane_of_item: dict[int, int] = {}
        for lane, items in allocation.items():
            for item in items:
                lane_of_item[item] = lane

        total_loss = 0.0
        for co_set in co_sets:
            lane_counts = [0] * num_lanes
            for item in co_set:
                lane = lane_of_item.get(item)
                if lane is not None and 0 <= lane < num_lanes:
                    lane_counts[lane] += 1
            if not any(lane_counts):
                continue
            base_lane = int(np.argmax(lane_counts))
            for item in co_set:
                lane = lane_of_item.get(item)
                if lane is None:
                    continue
                distance = abs(base_lane - lane)
                total_loss += self.loss_w1 * distance + self.loss_w2 * (distance**2)
        return total_loss

    def _extract_co_sets(self, order_data: list[dict], num_lanes: int) -> list[set[int]]:
        try:
            return self._extract_co_sets_with_gensim(order_data, num_lanes)
        except Exception:
            return self._extract_co_sets_fallback(order_data, num_lanes)

    def _extract_co_sets_with_gensim(self, order_data: list[dict], num_lanes: int) -> list[set[int]]:
        from gensim import corpora
        from gensim.models import LdaModel

        documents = []
        for order in order_data:
            sku_list = [f"sku_{int(sku)}" for sku in sorted(set(order["sku_list"]))]
            if len(sku_list) >= 2:
                documents.append(sku_list)

        if len(documents) < 10:
            return self._extract_co_sets_fallback(order_data, num_lanes)

        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]

        num_topics = max(2, min(self.lda_topics, num_lanes, len(dictionary)))
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=self.seed,
            passes=3,
            iterations=self.lda_iterations,
        )

        co_sets: list[set[int]] = []
        for topic_id in range(num_topics):
            topic_terms = model.show_topic(topic_id, topn=self.lda_top_n)
            sku_set: set[int] = set()
            for token, _weight in topic_terms:
                if token.startswith("sku_"):
                    sku_set.add(int(token.split("_", 1)[1]))
            if len(sku_set) >= 2:
                co_sets.append(sku_set)
        return co_sets

    def _extract_co_sets_fallback(self, order_data: list[dict], num_lanes: int) -> list[set[int]]:
        pair_counter: Counter[tuple[int, int]] = Counter()
        sku_counter: Counter[int] = Counter()

        for order in order_data:
            unique_skus = sorted(set(int(sku) for sku in order["sku_list"]))
            for sku in unique_skus:
                sku_counter[sku] += 1
            for i in range(len(unique_skus)):
                for j in range(i + 1, len(unique_skus)):
                    pair_counter[(unique_skus[i], unique_skus[j])] += 1

        adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (left, right), count in pair_counter.items():
            adjacency[left].append((right, count))
            adjacency[right].append((left, count))

        top_seeds = [sku for sku, _count in sku_counter.most_common(max(2, self.lda_topics))]
        co_sets: list[set[int]] = []
        for seed in top_seeds:
            neighbors = sorted(adjacency.get(seed, []), key=lambda x: x[1], reverse=True)
            sku_set = {seed}
            for neighbor, _count in neighbors[: self.lda_top_n - 1]:
                sku_set.add(neighbor)
            if len(sku_set) >= 2:
                co_sets.append(sku_set)

        return co_sets[: max(2, min(num_lanes, len(co_sets)))]

