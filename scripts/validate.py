#!/usr/bin/env python3
"""Run the full BL View pipeline against synthetic data and assert recovery.

Usage::

    python scripts/validate.py                       # generate + validate
    python scripts/validate.py --input data/samples/synthetic_cl31.dat
    python scripts/validate.py --json report.json

Exits non-zero if any injected layer is not recovered within its documented
tolerance, so this doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blview.config import Config                                    # noqa: E402
from blview.pipeline import run_pipeline                            # noqa: E402
from blview.synth.generator import (                                # noqa: E402
    SyntheticScenario, generate, write_truth, write_vaisala_file,
)
from blview.validate import (                                       # noqa: E402
    check, format_report, load_truth, score_all,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="raw file to validate (default: generate one)")
    p.add_argument("--truth", help="truth JSON (default: <input stem>_truth.json)")
    p.add_argument("--config", help="JSON config overriding the defaults")
    p.add_argument("--json", help="write the full report to this path")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--work-dir", default="data/validation",
        help="where generated data goes when --input is not given",
    )
    args = p.parse_args(argv)

    if args.input:
        raw = Path(args.input)
        truth_path = Path(args.truth) if args.truth else raw.with_name(
            raw.stem + "_truth.json"
        )
        if not truth_path.exists():
            print(f"error: truth file {truth_path} not found; validation needs the "
                  f"sidecar written alongside a synthetic file", file=sys.stderr)
            return 2
    else:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        raw = work / "synthetic_cl31.dat"
        truth_path = work / "synthetic_cl31_truth.json"
        overrides = {"duration_h": args.hours}
        if args.seed is not None:
            overrides["seed"] = args.seed
        scenario = SyntheticScenario(**overrides)
        print(f"generating {args.hours:.0f} h of synthetic data -> {raw}")
        data = generate(scenario)
        write_vaisala_file(raw, data, scenario)
        write_truth(truth_path, data)

    config = Config.load(args.config)
    print(f"running pipeline over {raw}")
    result = run_pipeline([raw], config=config)

    truth = load_truth(truth_path)
    report = score_all(truth, result.layers, result.processed.quality)
    print()
    print(format_report(report))

    failures = check(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"report": report, "failures": failures}, indent=2)
        )
        print(f"\nwrote {args.json}")

    print()
    if failures:
        print(f"FAILED ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED: every injected layer recovered within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
