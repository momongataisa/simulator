from __future__ import annotations

import random
from statistics import mean, pstdev
from typing import Iterable, Sequence

import numpy as np


Position = tuple[int, int]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def target_bucket_sizes(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        return []
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def chunk_evenly(items: Sequence[int], bucket_count: int) -> list[list[int]]:
    sizes = target_bucket_sizes(len(items), bucket_count)
    out: list[list[int]] = []
    start = 0
    for size in sizes:
        out.append(list(items[start : start + size]))
        start += size
    return out


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    value_list = list(values)
    if not value_list:
        return 0.0, 0.0
    if len(value_list) == 1:
        return float(value_list[0]), 0.0
    return float(mean(value_list)), float(pstdev(value_list))

