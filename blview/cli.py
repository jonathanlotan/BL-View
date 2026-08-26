"""``python -m blview.cli`` -- command line entry point for BL View."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import __version__


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON config file overriding blview.config.Config")
    p.add_argument(
        "--data-dir", default=None, help="override store.data_dir (default: data/)"
    )


def cmd_adapters(args: argparse.Namespace) -> int:
    from .adapters import available_adapters

    for name, cls in sorted(available_adapters().items()):
        print(f"{name:<14} {cls.description}")
        if cls.patterns:
            print(f"{'':<14} patterns: {', '.join(cls.patterns)}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth.generator import SyntheticScenario, generate, write_truth, write_vaisala_file

    overrides = {}
    if args.hours is not None:
        overrides["duration_h"] = args.hours
    if args.interval is not None:
        overrides["interval_s"] = args.interval
    if args.seed is not None:
        overrides["seed"] = args.seed
    scenario = SyntheticScenario(**overrides)

    data = generate(scenario, start_time=args.start)
    out = Path(args.output)
    write_vaisala_file(out, data, scenario)
    truth_path = out.with_name(out.stem + "_truth.json")
    write_truth(truth_path, data)

    n_screened = sum(s["screened"] for s in data["truth"])
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(data['truth'])} profiles x {scenario.n_gates} gates)")
    print(f"wrote {truth_path}  ({n_screened} profiles flagged fog/precipitation)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blview",
        description=(
            "BL View -- ceilometer boundary-layer, cloud and haze-layer viewer. "
            "All layer heights are aerosol-backscatter gradients, NOT thermodynamic "
            "measurements."
        ),
    )
    p.add_argument("--version", action="version", version=f"blview {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("adapters", help="list registered raw-format adapters")
    sp.set_defaults(func=cmd_adapters)

    sp = sub.add_parser("synth", help="generate a synthetic raw ceilometer file")
    sp.add_argument("-o", "--output", default="data/samples/synthetic_cl31.dat")
    sp.add_argument("--hours", type=float, default=None, help="duration (default 24)")
    sp.add_argument("--interval", type=float, default=None, help="profile interval, s")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument(
        "--start", type=float, default=None,
        help="start time, unix seconds (default: end the series at 'now')",
    )
    sp.set_defaults(func=cmd_synth)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
