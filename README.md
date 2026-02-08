# warehouse_simulator

Standalone warehouse throughput simulator built from the `AGENTS.md` specification.
This implementation is independent from the existing `my_simulator` code.

## Components

- `main.py`: Entry point
- `config.py`: Parameters
- `map.py`: Warehouse map and graph
- `order_generator.py`: Order generation
- `agent.py`: Agent state
- `task_manager.py`: Task generation and assignment
- `path_planner.py`: Priority-based A* with time-space reservations
- `simulator.py`: Discrete-time simulation loop
- `evaluator.py`: Multi-run statistics
- `strategies/`: Random / Frequency / CK / SL-LDA strategies

## Dependencies

```bash
pip install -r warehouse_simulator/requirements.txt
```

## Usage

Run all strategies:

```bash
python3 warehouse_simulator/main.py --method all --runs 3
```

Run one strategy:

```bash
python3 warehouse_simulator/main.py --method ck --runs 5
```
