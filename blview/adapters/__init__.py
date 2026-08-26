"""Raw-format adapters.

Importing this package registers every bundled adapter.  Third-party adapters
register themselves by importing :func:`register_adapter` and decorating their
class; point BL View at them with ``--adapter-module my.package``.
"""

from .base import (  # noqa: F401
    AdapterError,
    CeilometerAdapter,
    available_adapters,
    detect_adapter,
    get_adapter,
    iter_input_files,
    register_adapter,
)
from . import vaisala_cl  # noqa: F401  (registers VaisalaCLAdapter)
from . import generic_csv  # noqa: F401  (registers GenericCSVAdapter)

__all__ = [
    "AdapterError",
    "CeilometerAdapter",
    "available_adapters",
    "detect_adapter",
    "get_adapter",
    "iter_input_files",
    "register_adapter",
]
