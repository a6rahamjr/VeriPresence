from __future__ import annotations

import argparse
import json
from pathlib import Path

from veripresence.config import load_config
from veripresence.training.trainer import train
from veripresence.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VeriPresence identity model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    configure_logging()
    result = train(load_config(args.config))
    print(
        json.dumps(
            {
                "model_path": str(result.model_path),
                "run_dir": str(result.run_dir),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
