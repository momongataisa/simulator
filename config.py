from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WarehouseConfig:
    grid_height: int = 40
    grid_width: int = 25
    num_lanes: int = 25
    num_export_ports: int = 2
    num_agents: int = 25


@dataclass
class OrderConfig:
    num_skus: int = 250
    num_orders: int = 1200
    train_ratio: float = 0.8
    min_items_per_order: int = 1
    max_items_per_order: int = 4
    topic_count: int = 8
    co_occurrence_strength: float = 0.65
    popularity_skew: float = 1.1


@dataclass
class StrategyConfig:
    lda_topics: int = 8
    lda_top_n: int = 12
    lda_iterations: int = 150
    sa_iterations: int = 1800
    sa_initial_temp: float = 40.0
    sa_cooling_rate: float = 0.996
    loss_w1: float = 1.0
    loss_w2: float = 1.8


@dataclass
class SimulationConfig:
    max_steps: int = 3000
    plan_horizon: int = 120
    pickup_wait_steps: int = 1
    seed: int = 42

