"""Compare pipeline output against injected synthetic ground truth.

The synthetic generator records what it injected at every timestamp, together
with whether that feature was **observable** -- a ceilometer cannot see through
cloud, and fog/precipitation profiles are screened before detection runs, so
asserting recovery there would be demanding that the tool invent data
(ASSUMPTIONS.md #S4).  Scoring is therefore restricted to observable
timestamps, and the counts of skipped timestamps are reported so the
restriction is visible rather than hidden.

Reported per feature:

``detection_rate``
    fraction of observable timestamps where a layer of the right type was
    reported at all;
``bias`` / ``mae`` / ``rmse`` / ``p95``
    signed and unsigned height errors, metres;
``within_tolerance``
    fraction of matched timestamps inside the documented tolerance;
``false_positive_rate``
    fraction of timestamps where the feature was reported but truth says it
    was absent (and the profile was not screened).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .model import Layer, LayerType

#: Height tolerances, metres.  Justified in README "Validation".
DEFAULT_TOLERANCES: dict[str, float] = {
    "cloud_base": 50.0,       # sharp edge, essentially gate-resolution limited
    "sml_top": 150.0,         # ~1 gate of entrainment-zone depth
    "rl_top": 150.0,
    "haze_base": 150.0,
    "haze_top": 200.0,        # the weakest edge in the scene
}

#: Minimum fraction of observable timestamps that must be detected.
DEFAULT_MIN_DETECTION_RATE: dict[str, float] = {
    "cloud_base": 0.90,
    "sml_top": 0.85,
    "rl_top": 0.75,           # genuinely ambiguous during the two transitions
    "haze_base": 0.80,
    "haze_top": 0.80,
}

#: Maximum tolerated |mean signed error|, metres.
DEFAULT_MAX_BIAS = 75.0
#: Maximum tolerated rate of reporting a feature that was not there.
DEFAULT_MAX_FALSE_POSITIVE_RATE = 0.15
#: Minimum fraction of matched detections that must land inside the tolerance.
#: Measured across seven synthetic days at different diurnal phases and noise
#: seeds, the mixing-layer top -- the weakest feature, because the mixing and
#: residual layers genuinely merge during the two transitions -- ranges 89.9%
#: to 91.5%.  The gate is set below that range on purpose: it is a regression
#: guard, and a gate tuned to the best observed run fails on the next one for
#: reasons that have nothing to do with the code.
DEFAULT_MIN_WITHIN_TOLERANCE = 0.85


@dataclass
class FeatureScore:
    """Scores for one truth feature."""

    feature: str
    n_observable: int
    n_matched: int
    n_absent: int
    n_false_positive: int
    detection_rate: float
    false_positive_rate: float
    bias: float
    mae: float
    rmse: float
    p95: float
    within_tolerance: float
    tolerance_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScreeningScore:
    """Precision/recall of the fog and precipitation screens."""

    name: str
    n_truth: int
    n_flagged: int
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def recall(self) -> float:
        return self.true_positive / self.n_truth if self.n_truth else 1.0

    @property
    def precision(self) -> float:
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total else 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recall"] = self.recall
        d["precision"] = self.precision
        return d


def load_truth(path: str | Path) -> list[dict[str, Any]]:
    """Load the ``*_truth.json`` sidecar written by the generator."""
    payload = json.loads(Path(path).read_text())
    return payload["states"]


# --------------------------------------------------------------- extraction
def _best(layers: list[Layer], layer_type: LayerType) -> Optional[Layer]:
    """Highest-confidence layer of a given type in one profile."""
    matches = [l for l in layers if l.type is layer_type]
    if not matches:
        return None
    return max(matches, key=lambda l: l.confidence)


def _lowest(layers: list[Layer], layer_type: LayerType) -> Optional[Layer]:
    """Lowest layer of a given type -- used for cloud, where the base that
    matters is the first one the beam meets."""
    matches = [l for l in layers if l.type is layer_type]
    if not matches:
        return None
    return min(matches, key=lambda l: l.base_height)


#: feature name -> (truth key, visibility key, extractor from one profile)
FEATURES: dict[str, tuple[str, str, Callable[[list[Layer]], Optional[float]]]] = {
    "cloud_base": (
        "cloud_base", "cloud_visible",
        lambda ls: (lambda l: l.base_height if l else None)(_lowest(ls, LayerType.CLOUD)),
    ),
    "sml_top": (
        "sml_top", "sml_visible",
        lambda ls: (lambda l: l.top_height if l else None)(_best(ls, LayerType.MIXING_LAYER)),
    ),
    "rl_top": (
        "rl_top", "rl_visible",
        lambda ls: (lambda l: l.top_height if l else None)(_best(ls, LayerType.RESIDUAL_LAYER)),
    ),
    "haze_base": (
        "haze_base", "haze_visible",
        lambda ls: (lambda l: l.base_height if l else None)(_best(ls, LayerType.HAZE)),
    ),
    "haze_top": (
        "haze_top", "haze_visible",
        lambda ls: (lambda l: l.top_height if l else None)(_best(ls, LayerType.HAZE)),
    ),
}


def score_feature(
    feature: str,
    truth: list[dict[str, Any]],
    layers: list[list[Layer]],
    tolerance_m: float,
) -> FeatureScore:
    """Score one feature over the whole run."""
    truth_key, visible_key, extract = FEATURES[feature]
    errors: list[float] = []
    n_observable = n_matched = n_absent = n_false_positive = 0

    for state, profile_layers in zip(truth, layers):
        expected = state.get(truth_key)
        observable = bool(state.get(visible_key))
        detected = extract(profile_layers)

        if observable and expected is not None:
            n_observable += 1
            if detected is not None:
                n_matched += 1
                errors.append(float(detected) - float(expected))
        elif not state.get("screened") and expected is None:
            # Truth says the feature is absent and the profile was usable:
            # anything reported here is a false positive.
            n_absent += 1
            if detected is not None:
                n_false_positive += 1

    err = np.array(errors) if errors else np.array([])
    absolute = np.abs(err)
    return FeatureScore(
        feature=feature,
        n_observable=n_observable,
        n_matched=n_matched,
        n_absent=n_absent,
        n_false_positive=n_false_positive,
        detection_rate=(n_matched / n_observable) if n_observable else float("nan"),
        false_positive_rate=(n_false_positive / n_absent) if n_absent else 0.0,
        bias=float(err.mean()) if err.size else float("nan"),
        mae=float(absolute.mean()) if err.size else float("nan"),
        rmse=float(np.sqrt((err ** 2).mean())) if err.size else float("nan"),
        p95=float(np.percentile(absolute, 95)) if err.size else float("nan"),
        within_tolerance=(
            float((absolute <= tolerance_m).mean()) if err.size else float("nan")
        ),
        tolerance_m=tolerance_m,
    )


def score_screening(
    truth: list[dict[str, Any]], quality: np.ndarray
) -> list[ScreeningScore]:
    """Score the fog and precipitation screens against truth."""
    from .model import QualityFlag

    out = []
    for name, truth_key, flag in (
        ("fog", "fog", QualityFlag.FOG),
        ("precipitation", "precip", QualityFlag.PRECIPITATION),
    ):
        expected = np.array([bool(s[truth_key]) for s in truth])
        flagged = (np.asarray(quality) & int(flag)) != 0
        out.append(
            ScreeningScore(
                name=name,
                n_truth=int(expected.sum()),
                n_flagged=int(flagged.sum()),
                true_positive=int((flagged & expected).sum()),
                false_positive=int((flagged & ~expected).sum()),
                false_negative=int((~flagged & expected).sum()),
            )
        )
    return out


def score_all(
    truth: list[dict[str, Any]],
    layers: list[list[Layer]],
    quality: np.ndarray,
    tolerances: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Score every feature and both screens."""
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    n = min(len(truth), len(layers))
    truth, layers = truth[:n], layers[:n]
    return {
        "n_profiles": n,
        "features": {
            name: score_feature(name, truth, layers, tolerances[name]).to_dict()
            for name in FEATURES
        },
        "screening": [s.to_dict() for s in score_screening(truth, quality[:n])],
    }


def check(
    report: dict[str, Any],
    min_detection_rate: Optional[dict[str, float]] = None,
    max_bias: float = DEFAULT_MAX_BIAS,
    max_false_positive_rate: float = DEFAULT_MAX_FALSE_POSITIVE_RATE,
    min_within_tolerance: float = DEFAULT_MIN_WITHIN_TOLERANCE,
) -> list[str]:
    """Return a list of failure messages; empty means every check passed."""
    thresholds = {**DEFAULT_MIN_DETECTION_RATE, **(min_detection_rate or {})}
    failures: list[str] = []

    for name, score in report["features"].items():
        required = thresholds.get(name, 0.8)
        if score["n_observable"] == 0:
            failures.append(f"{name}: no observable timestamps to score against")
            continue
        if score["detection_rate"] < required:
            failures.append(
                f"{name}: detection rate {score['detection_rate']:.1%} "
                f"below required {required:.0%}"
            )
        if not np.isfinite(score["bias"]) or abs(score["bias"]) > max_bias:
            failures.append(
                f"{name}: height bias {score['bias']:.1f} m exceeds +/-{max_bias:.0f} m"
            )
        if score["within_tolerance"] < min_within_tolerance:
            failures.append(
                f"{name}: only {score['within_tolerance']:.1%} of detections within "
                f"+/-{score['tolerance_m']:.0f} m (need {min_within_tolerance:.0%})"
            )
        if score["false_positive_rate"] > max_false_positive_rate:
            failures.append(
                f"{name}: false-positive rate {score['false_positive_rate']:.1%} "
                f"exceeds {max_false_positive_rate:.0%}"
            )

    for screen in report["screening"]:
        if screen["recall"] < 0.95:
            failures.append(
                f"screening/{screen['name']}: recall {screen['recall']:.1%} below 95%"
            )
        if screen["precision"] < 0.80:
            failures.append(
                f"screening/{screen['name']}: precision {screen['precision']:.1%} below 80%"
            )
    return failures


def format_report(report: dict[str, Any]) -> str:
    """Human-readable summary table."""
    lines = [
        f"Profiles scored: {report['n_profiles']}",
        "",
        f"{'feature':<12} {'obs':>5} {'det':>5} {'rate':>7} {'bias':>8} {'MAE':>7} "
        f"{'RMSE':>7} {'p95':>7} {'<=tol':>7} {'FP':>7}",
        "-" * 82,
    ]
    for name, s in report["features"].items():
        lines.append(
            f"{name:<12} {s['n_observable']:>5} {s['n_matched']:>5} "
            f"{s['detection_rate']:>6.1%} {s['bias']:>7.1f}m {s['mae']:>6.1f}m "
            f"{s['rmse']:>6.1f}m {s['p95']:>6.1f}m {s['within_tolerance']:>6.1%} "
            f"{s['false_positive_rate']:>6.1%}"
        )
    lines += ["", f"{'screen':<16} {'truth':>6} {'flagged':>8} {'recall':>8} {'precision':>10}", "-" * 52]
    for s in report["screening"]:
        lines.append(
            f"{s['name']:<16} {s['n_truth']:>6} {s['n_flagged']:>8} "
            f"{s['recall']:>7.1%} {s['precision']:>9.1%}"
        )
    return "\n".join(lines)
