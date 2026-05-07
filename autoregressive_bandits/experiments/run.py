from __future__ import annotations

import argparse
import json

from autoregressive_bandits.experiments.config import load_config
from autoregressive_bandits.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an autoregressive bandit comparison experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML scenario config.")
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()
    summary = run_experiment(load_config(args.config), args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
