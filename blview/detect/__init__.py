"""Multi-layer detection: Haar covariance edges, classification, tracking."""

from .detector import detect_layers  # noqa: F401
from .tracking import track_layers, tracking_height  # noqa: F401
from .haar import (  # noqa: F401
    haar_covariance,
    haar_noise,
    half_width_gates,
    local_extrema,
    log_field,
    multiscale,
)
from .layers import (  # noqa: F401
    CloudDetection,
    EdgeCandidate,
    aggregate_edges,
    classify_profile,
    detect_clouds,
    refine_edge_heights,
)

__all__ = [
    "detect_layers",
    "track_layers",
    "tracking_height",
    "haar_covariance",
    "haar_noise",
    "half_width_gates",
    "local_extrema",
    "log_field",
    "multiscale",
    "CloudDetection",
    "EdgeCandidate",
    "aggregate_edges",
    "classify_profile",
    "detect_clouds",
    "refine_edge_heights",
]
