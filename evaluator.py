from __future__ import annotations

from dataclasses import dataclass

from warehouse_simulator.config import OrderConfig, SimulationConfig, WarehouseConfig
from warehouse_simulator.simulator import Simulator
from warehouse_simulator.utils import mean_std


@dataclass
class EvaluationRow:
    method: str
    mean: float
    std: float
    relative_to_random: float


class Evaluator:
    def __init__(
        self,
        warehouse_config: WarehouseConfig,
        order_config: OrderConfig,
        simulation_config: SimulationConfig,
    ):
        self.warehouse_config = warehouse_config
        self.order_config = order_config
        self.simulation_config = simulation_config

    def evaluate(
        self,
        strategies: list,
        runs: int = 3,
        base_seed: int = 42,
    ) -> list[EvaluationRow]:
        throughput_logs: dict[str, list[float]] = {strategy.name: [] for strategy in strategies}

        for run_index in range(runs):
            run_seed = base_seed + run_index * 1000
            simulator = Simulator(
                warehouse_config=self.warehouse_config,
                order_config=self.order_config,
                simulation_config=self.simulation_config,
            )

            for strategy in strategies:
                result = simulator.run(strategy, seed=run_seed)
                throughput_logs[strategy.name].append(result.throughput)

        random_mean = 1.0
        if "Random" in throughput_logs:
            random_mean, _ = mean_std(throughput_logs["Random"])
            random_mean = max(1e-9, random_mean)
        elif strategies:
            fallback_name = strategies[0].name
            random_mean, _ = mean_std(throughput_logs[fallback_name])
            random_mean = max(1e-9, random_mean)

        rows: list[EvaluationRow] = []
        for strategy in strategies:
            method = strategy.name
            mean_value, std_value = mean_std(throughput_logs[method])
            rows.append(
                EvaluationRow(
                    method=method,
                    mean=mean_value,
                    std=std_value,
                    relative_to_random=mean_value / random_mean,
                )
            )
        return rows

    @staticmethod
    def format_table(rows: list[EvaluationRow]) -> str:
        headers = ["Method", "Mean", "Std", "Relative(Random=1.0)"]
        lines = []
        lines.append(f"{headers[0]:<12} {headers[1]:>10} {headers[2]:>10} {headers[3]:>20}")
        lines.append("-" * 58)
        for row in rows:
            lines.append(
                f"{row.method:<12} {row.mean:>10.4f} {row.std:>10.4f} {row.relative_to_random:>20.4f}"
            )
        return "\n".join(lines)

