"""``python -m edge_hub`` entry point."""

from __future__ import annotations

import argparse
import asyncio
import contextlib

from .app import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="edge-security-hub")
    parser.add_argument("--data-dir", default=None, help="state directory (default /data)")
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args.data_dir))


if __name__ == "__main__":
    main()
