"""Data ingestion, validation, and warehouse loaders.

Submodules
----------
``models``     SQLAlchemy ORM definitions (source of truth for the DB schema).
``sources``    Thin clients over external data providers + a raw landing zone.
``loaders``    Idempotent upserts into the ``warehouse`` schema.
``crosswalk``  Player-ID reconciliation via the Chadwick Bureau register.
"""

__version__ = "0.0.0"
