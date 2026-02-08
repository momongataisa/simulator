from __future__ import annotations

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from warehouse_simulator.config import (  # noqa: E402
    OrderConfig,
    SimulationConfig,
    StrategyConfig,
    WarehouseConfig,
)
from warehouse_simulator.evaluator import Evaluator  # noqa: E402
from warehouse_simulator.strategies import (  # noqa: E402
    CKStrategy,
    FrequencyStrategy,
    RandomStrategy,
    SLLDAStrategy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warehouse throughput simulator")
    parser.add_argument(
        "--method",
        choices=["all", "random", "frequency", "ck", "sl-lda"],
        default="all",
        help="Storage location strategy to evaluate",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for each method")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--max-steps", type=int, default=3000, help="Maximum simulation steps")
    parser.add_argument("--num-orders", type=int, default=1200, help="Number of generated orders")
    parser.add_argument("--num-skus", type=int, default=250, help="Number of SKUs")
    parser.add_argument("--lanes", type=int, default=25, help="Number of lanes")
    parser.add_argument("--ports", type=int, default=2, help="Number of export ports")
    parser.add_argument("--agents", type=int, default=25, help="Number of agents")
    parser.add_argument("--grid-height", type=int, default=40, help="Grid height")
    parser.add_argument("--grid-width", type=int, default=25, help="Grid width")
    return parser


def build_strategies(method: str, seed: int, strategy_config: StrategyConfig):
    all_strategies = {
        "random": RandomStrategy(seed=seed),
        "frequency": FrequencyStrategy(),
        "ck": CKStrategy(seed=seed),
        "sl-lda": SLLDAStrategy(
            seed=seed,
            lda_topics=strategy_config.lda_topics,
            lda_top_n=strategy_config.lda_top_n,
            lda_iterations=strategy_config.lda_iterations,
            sa_iterations=strategy_config.sa_iterations,
            sa_initial_temp=strategy_config.sa_initial_temp,
            sa_cooling_rate=strategy_config.sa_cooling_rate,
            loss_w1=strategy_config.loss_w1,
            loss_w2=strategy_config.loss_w2,
        ),
    }
    if method == "all":
        return [
            all_strategies["random"],
            all_strategies["frequency"],
            all_strategies["ck"],
            all_strategies["sl-lda"],
        ]
    return [all_strategies[method]]


def main() -> None:
    args = build_parser().parse_args()

    warehouse_config = WarehouseConfig(
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        num_lanes=args.lanes,
        num_export_ports=args.ports,
        num_agents=args.agents,
    )
    order_config = OrderConfig(
        num_skus=args.num_skus,
        num_orders=args.num_orders,
    )
    strategy_config = StrategyConfig()
    simulation_config = SimulationConfig(
        max_steps=args.max_steps,
        seed=args.seed,
    )

    strategies = build_strategies(args.method, args.seed, strategy_config)
    evaluator = Evaluator(warehouse_config, order_config, simulation_config)
    rows = evaluator.evaluate(strategies=strategies, runs=args.runs, base_seed=args.seed)
    print(Evaluator.format_table(rows))


if __name__ == "__main__":
    main()

