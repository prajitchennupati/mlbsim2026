"""Read-only HTTP API over the ``serving`` schema.

The API never runs a model: it reads immutable, versioned prediction rows the
pipeline has already written. Domain routers arrive in M7; M0 ships liveness
and version endpoints only.
"""

__version__ = "0.0.0"
