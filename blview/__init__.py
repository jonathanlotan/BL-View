"""BL View -- aerosol-backscatter boundary-layer, cloud and haze-layer viewer.

BL View ingests raw ceilometer attenuated-backscatter profiles, preprocesses
them, detects multiple aerosol/cloud layer boundaries per profile, tracks those
layers in time, and serves the result to a browser quicklook.

IMPORTANT SCIENTIFIC CAVEAT
---------------------------
Every "layer boundary" produced by this package is an **aerosol backscatter
gradient**, not a thermodynamic measurement.  BL View does not measure, derive
or estimate temperature, potential temperature, or inversion strength.  Where
the term "inversion proxy" is used it means exactly this: aerosol layering is
often *co-located* with a thermodynamic inversion, and is used as an indirect
indicator of it.  It is not a substitute for a radiosonde, a microwave
radiometer, or a mast.
"""

__version__ = "0.1.0"

UNITS_BETA = "m-1 sr-1"      #: internal attenuated-backscatter units
UNITS_RANGE = "m"            #: internal range/height units (above instrument)
UNITS_TIME = "seconds since 1970-01-01T00:00:00Z"  #: internal time units
