"""Offline field test for the bundled WS3 scenario inventory/products report."""

import json
from dataclasses import asdict
from pathlib import Path

import ws3.agent


MODEL_DIR = Path(__file__).parent / 'data' / 'woodstock_model_files_tsa24_clipped'
MODEL_NAME = 'tsa24_clipped'


def main() -> int:
    result = ws3.agent.report_scenario_inventory_products(MODEL_DIR, MODEL_NAME)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
