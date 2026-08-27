"""``python -m blview.cli`` -- command line entry point for BL View.

    blview adapters                 list registered raw-format readers
    blview synth   -o FILE          generate a synthetic raw ceilometer file
    blview ingest  FILE...          run the pipeline and store the result
    blview status                   what is currently in the store
    blview validate                 run the validation harness
    blview serve                    serve the API and the quicklook page
    blview demo                     generate + ingest + validate + serve
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Config

log = logging.getLogger("blview")


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(getattr(args, "config", None))
    if getattr(args, "data_dir", None):
        config.store.data_dir = Path(args.data_dir)
    return config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON file overriding blview.config.Config")
    parser.add_argument("--data-dir", help="store location (default: data/)")


def _stamp(value: float | None) -> str:
    if value is None:
        return "-"
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


# --------------------------------------------------------------- commands --
def cmd_adapters(args: argparse.Namespace) -> int:
    from .adapters import available_adapters

    for name, cls in sorted(available_adapters().items()):
        print(f"{name:<14} {cls.description}")
        if cls.patterns:
            print(f"{'':<14} patterns: {', '.join(cls.patterns)}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth.generator import (
        SyntheticScenario, generate, write_truth, write_vaisala_file,
    )

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

    screened = sum(s["screened"] for s in data["truth"])
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(data['truth'])} profiles x {scenario.n_gates} gates)")
    print(f"wrote {truth_path}  ({screened} profiles flagged fog/precipitation)")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline
    from .store import Store

    config = _load_config(args)
    options = json.loads(args.adapter_options) if args.adapter_options else None
    result = run_pipeline(
        args.inputs, config=config, adapter=args.adapter, adapter_options=options
    )
    info = Store(config).write(result.processed, result.layers)

    counts: dict[str, int] = {}
    for layer in result.all_layers:
        counts[str(layer.type)] = counts.get(str(layer.type), 0) + 1
    screened = int(result.processed.screened_mask().sum())

    print(f"adapter        : {result.adapter_name}")
    print(f"profiles       : {info['profiles']} "
          f"({screened} screened as precipitation/fog/low-SNR)")
    print(f"time span      : {_stamp(float(result.processed.time[0]))} .. "
          f"{_stamp(float(result.processed.time[-1]))}")
    print(f"layers stored  : {info['layers']}")
    for name in sorted(counts):
        print(f"  {name:<16} {counts[name]}")
    print(f"grid files     : {len(info['grid_files'])} written, "
          f"{info['purged_files']} purged by retention")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .store import Store

    info = Store(_load_config(args)).status()
    if info["time_end"] is None:
        print("store is empty -- run `blview ingest` or `blview demo`")
        return 0
    print(f"site           : {info['site_name']}")
    print(f"instrument     : {info['instrument']}")
    print(f"data available : {_stamp(info['time_start'])} .. {_stamp(info['time_end'])}"
          f"  ({info['hours_available']:.1f} h, retention {info['retention_hours']:.0f} h)")
    lowest = info["lowest_usable_height_m"]
    print(f"vertical grid  : {info['range_resolution_m']:.0f} m gates up to "
          f"{info['max_range_m']:.0f} m; lowest usable "
          f"{lowest:.0f} m" if lowest else "vertical grid  : unknown")
    print(f"on disk        : {info['n_grid_files']} grid files, "
          f"{info['grid_bytes'] / 1e6:.1f} MB + {info['database_bytes'] / 1e6:.1f} MB index")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.validate import main as validate_main

    argv = []
    if args.input:
        argv += ["--input", args.input]
    if args.config:
        argv += ["--config", args.config]
    if args.json:
        argv += ["--json", args.json]
    argv += ["--hours", str(args.hours)]
    return validate_main(argv)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    config = _load_config(args)
    app = create_app(config)
    print(f"BL View {__version__} serving on http://{args.host}:{args.port}")
    print("  quicklook   http://%s:%s/" % (args.host, args.port))
    print("  API docs    http://%s:%s/docs" % (args.host, args.port))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate synthetic data, run the pipeline, validate it, then serve."""
    from .pipeline import run_pipeline
    from .store import Store
    from .synth.generator import (
        SyntheticScenario, generate, write_truth, write_vaisala_file,
    )
    from .validate import check, format_report, load_truth, score_all

    config = _load_config(args)
    # Generated data goes to data/generated/, which is git-ignored. The small
    # committed sample under data/samples/ is documentation, not scratch space.
    generated = Path(config.store.data_dir) / "generated"
    raw = generated / "synthetic_cl31.dat"
    truth_path = generated / "synthetic_cl31_truth.json"

    if args.regenerate or not raw.exists():
        print(f"[1/4] generating {args.hours:.0f} h of synthetic CL31 data ...")
        scenario = SyntheticScenario(duration_h=args.hours)
        data = generate(scenario)
        write_vaisala_file(raw, data, scenario)
        write_truth(truth_path, data)
        print(f"      {raw}  ({raw.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[1/4] using existing {raw} (pass --regenerate to rebuild)")

    print("[2/4] ingest -> preprocess -> detect -> track ...")
    result = run_pipeline([raw], config=config)
    info = Store(config).write(result.processed, result.layers)
    print(f"      {info['profiles']} profiles, {info['layers']} layers, "
          f"{len(info['grid_files'])} grid files")

    print("[3/4] validating against the injected truth ...")
    report = score_all(load_truth(truth_path), result.layers, result.processed.quality)
    print()
    print(format_report(report))
    print()
    failures = check(report)
    if failures:
        print(f"      VALIDATION FAILED ({len(failures)}):")
        for failure in failures:
            print(f"        - {failure}")
        if not args.ignore_validation:
            return 1
    else:
        print("      validation passed: every injected layer recovered within tolerance.")

    if args.no_serve:
        return 0
    print(f"\n[4/4] serving ...")
    return cmd_serve(args)


# ----------------------------------------------------------------- parser --
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blview",
        description=(
            "BL View -- ceilometer boundary-layer, cloud and haze-layer viewer. "
            "Every layer height reported is an aerosol backscatter gradient, NOT a "
            "thermodynamic measurement."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"blview {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log each pipeline stage"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("adapters", help="list registered raw-format adapters")
    p.set_defaults(func=cmd_adapters)

    p = sub.add_parser("synth", help="generate a synthetic raw ceilometer file")
    p.add_argument("-o", "--output", default="data/generated/synthetic_cl31.dat")
    p.add_argument("--hours", type=float, default=None, help="duration (default 24)")
    p.add_argument("--interval", type=float, default=None, help="profile interval, s")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--start", type=float, default=None,
                   help="start time, unix seconds (default: end the series at now)")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("ingest", help="run the pipeline over raw files and store it")
    p.add_argument("inputs", nargs="+", help="raw files, directories or globs")
    p.add_argument("--adapter", help="force an adapter (default: auto-detect)")
    p.add_argument("--adapter-options", help="JSON dict passed to the adapter")
    _add_common(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("status", help="summarise what is in the store")
    _add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("validate", help="run the validation harness")
    p.add_argument("--input", help="raw file (default: generate one)")
    p.add_argument("--json", help="write the full report here")
    p.add_argument("--hours", type=float, default=24.0)
    _add_common(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("serve", help="serve the API and quicklook page")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    _add_common(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="generate + ingest + validate + serve")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="warning")
    p.add_argument("--regenerate", action="store_true",
                   help="rebuild the synthetic file even if it already exists")
    p.add_argument("--no-serve", action="store_true",
                   help="stop after validating instead of starting the server")
    p.add_argument("--ignore-validation", action="store_true",
                   help="serve even if validation fails")
    _add_common(p)
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
