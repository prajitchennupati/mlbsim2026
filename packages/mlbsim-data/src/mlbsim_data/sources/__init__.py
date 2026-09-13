"""Thin clients over external data providers.

Every client writes the verbatim response to the raw landing zone
(``settings.landing_dir``) before any parsing, so ingestion is replayable and
auditable.
"""
