"""Utilities for generating messages and inspecting hash execution."""

from .generator import CHARACTER_GROUPS, generate_bytes, generate_message, select_characters
from .dataset import (
    DigestRecord,
    SourceSpec,
    build_digest_records,
    generate_source_messages,
    select_digest_representatives,
    split_digest_groups,
    split_validation_report,
)
from .runner import ExperimentConfig, run_experiment
from .sanity import exhaustive_sanity_search
from .validation import validate_confirmatory_runs, write_confirmatory_validation

__all__ = [
    "CHARACTER_GROUPS",
    "DigestRecord",
    "ExperimentConfig",
    "SourceSpec",
    "build_digest_records",
    "generate_bytes",
    "generate_message",
    "generate_source_messages",
    "exhaustive_sanity_search",
    "run_experiment",
    "select_characters",
    "select_digest_representatives",
    "split_digest_groups",
    "split_validation_report",
    "validate_confirmatory_runs",
    "write_confirmatory_validation",
]
