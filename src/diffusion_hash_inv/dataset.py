"""Deterministic source data, digest targets, and leakage-safe splits."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, Mapping, Sequence

from .encoding.cgge import PRINTABLE94


SourceDistribution = Literal["printable", "random_bytes"]


def normalize_algorithm(algorithm: str) -> str:
    """Return the two experiment algorithms in hashlib's canonical spelling."""
    normalized = algorithm.lower().replace("-", "").replace("_", "")
    if normalized not in {"md5", "sha256"}:
        raise ValueError("algorithm must be md5 or sha256")
    return normalized


def digest_prefix_hex(digest: bytes, q: int) -> str:
    """Return exactly the first q bits as a zero-padded hexadecimal key."""
    if not 1 <= q <= len(digest) * 8:
        raise ValueError("q must be within the digest length")
    byte_count = (q + 7) // 8
    value = int.from_bytes(digest[:byte_count], "big") >> (byte_count * 8 - q)
    return f"{value:0{(q + 3) // 4}x}"


def hash_caption(algorithm: str, q: int, digest: bytes, *, length: int | None = None) -> str:
    """Build the fixed text condition used by image experiments."""
    caption = f"{normalize_algorithm(algorithm)}-{q}:{digest_prefix_hex(digest, q)}"
    return caption if length is None else f"{caption}|len_bytes={length}"


@dataclass(frozen=True)
class SourceSpec:
    distribution: SourceDistribution
    size: int
    seed: int = 0
    min_length: int = 4
    max_length: int = 31

    def __post_init__(self) -> None:
        if self.distribution not in {"printable", "random_bytes"}:
            raise ValueError("unsupported source distribution")
        if self.size < 1 or self.min_length < 0 or self.min_length > self.max_length:
            raise ValueError("invalid source size or length range")


@dataclass(frozen=True)
class DigestRecord:
    id: int
    source: SourceDistribution
    message: bytes
    algorithm: str
    q: int
    digest: bytes

    @property
    def prefix(self) -> str:
        return digest_prefix_hex(self.digest, self.q)

    @property
    def caption(self) -> str:
        return hash_caption(self.algorithm, self.q, self.digest)

    def to_json(self, split: str) -> dict[str, object]:
        return {
            "id": self.id,
            "split": split,
            "source": self.source,
            "message_hex": self.message.hex(),
            "length": len(self.message),
            "algorithm": self.algorithm,
            "q": self.q,
            "full_digest": self.digest.hex(),
            "prefix": self.prefix,
            "caption": self.caption,
        }


def generate_source_messages(spec: SourceSpec) -> tuple[bytes, ...]:
    """Generate a unique, deterministic source corpus without storing RNG state."""
    alphabet_size = 94 if spec.distribution == "printable" else 256
    if spec.size > sum(alphabet_size**length for length in range(spec.min_length, spec.max_length + 1)):
        raise ValueError("requested unique count exceeds source domain")
    rng = random.Random(spec.seed)
    messages: set[bytes] = set()
    alphabet = PRINTABLE94.encode("ascii")
    while len(messages) < spec.size:
        length = rng.randint(spec.min_length, spec.max_length)
        if spec.distribution == "printable":
            message = bytes(alphabet[rng.randrange(len(alphabet))] for _ in range(length))
        else:
            message = rng.randbytes(length)
        messages.add(message)
    return tuple(sorted(messages))


def build_digest_records(
    messages: Sequence[bytes], *, source: SourceDistribution, algorithm: str, q: int
) -> tuple[DigestRecord, ...]:
    """Attach one full digest and q-bit condition to each source message."""
    algorithm = normalize_algorithm(algorithm)
    digest_prefix_hex(hashlib.new(algorithm).digest(), q)
    records = []
    for identifier, message in enumerate(messages):
        digest = hashlib.new(algorithm, message).digest()
        records.append(DigestRecord(identifier, source, message, algorithm, q, digest))
    return tuple(records)


def split_digest_groups(
    records: Sequence[DigestRecord], *, seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Mapping[str, tuple[DigestRecord, ...]]:
    """Split whole truncated-digest groups so a condition never crosses splits."""
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("ratios must be three positive values summing to one")
    groups: dict[str, list[DigestRecord]] = {}
    for record in records:
        groups.setdefault(record.prefix, []).append(record)
    if records and any((record.algorithm, record.q) != (records[0].algorithm, records[0].q) for record in records):
        raise ValueError("all records in one split must share algorithm and q")
    names = ("train", "validation", "test")
    targets = [len(records) * ratio for ratio in ratios]
    counts = [0, 0, 0]
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    result: dict[str, list[DigestRecord]] = {name: [] for name in names}
    for _, group in items:
        index = max(range(3), key=lambda candidate: (targets[candidate] - counts[candidate], -candidate))
        result[names[index]].extend(group)
        counts[index] += len(group)
    return {name: tuple(result[name]) for name in names}


def select_digest_representatives(records: Sequence[DigestRecord], *, seed: int | None = None) -> tuple[DigestRecord, ...]:
    """Choose one deterministic source record for every q-bit digest target."""
    representatives: dict[str, DigestRecord] = {}
    def priority(record):
        return record.id if seed is None else hashlib.sha256(f"{seed}:".encode() + record.message).digest()
    for record in records:
        current = representatives.get(record.prefix)
        if current is None or priority(record) < priority(current):
            representatives[record.prefix] = record
    return tuple(representatives[prefix] for prefix in sorted(representatives))


def split_validation_report(split: Mapping[str, Sequence[DigestRecord]]) -> dict[str, object]:
    """Return the G0 overlap audit for a one-algorithm, one-q split."""
    names = ("train", "validation", "test")
    partitions = {name: tuple(split.get(name, ())) for name in names}
    pairs = {}
    passed = True
    for left, right in combinations(names, 2):
        messages = {record.message for record in partitions[left]} & {record.message for record in partitions[right]}
        digests = {record.prefix for record in partitions[left]} & {record.prefix for record in partitions[right]}
        passed &= not messages and not digests
        pairs[f"{left}_{right}"] = {
            "message_overlap_count": len(messages),
            "message_examples_hex": [message.hex() for message in sorted(messages)[:3]],
            "digest_overlap_count": len(digests),
            "digest_examples": sorted(digests)[:3],
        }
    settings = {(record.algorithm, record.q) for records in partitions.values() for record in records}
    return {
        "passed": passed and len(settings) <= 1,
        "algorithm_q_consistent": len(settings) <= 1,
        "pairwise": pairs,
        "record_counts": {name: len(records) for name, records in partitions.items()},
        "unique_digest_counts": {name: len({record.prefix for record in records}) for name, records in partitions.items()},
    }


def write_split(
    split: Mapping[str, Sequence[DigestRecord]], output_dir: str | Path, *, source_spec: SourceSpec, split_seed: int
) -> None:
    """Persist one JSONL data artifact and its minimal reproducibility manifest."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "records.jsonl").open("w", encoding="utf-8") as output:
        for name in ("train", "validation", "test"):
            for record in split.get(name, ()):
                output.write(json.dumps(record.to_json(name), sort_keys=True) + "\n")
    first = next((record for records in split.values() for record in records), None)
    if first is None:
        raise ValueError("split must contain at least one record")
    if any((record.algorithm, record.q) != (first.algorithm, first.q) for records in split.values() for record in records):
        raise ValueError("all records in one artifact must share algorithm and q")
    manifest = {
        "source": asdict(source_spec),
        "split_seed": split_seed,
        "algorithm": first.algorithm,
        "q": first.q,
        "counts": {name: len(records) for name, records in split.items()},
        "caption_format": "<algorithm>-<q>:<hex digest>",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quota_digest_split(spec: SourceSpec, *, algorithm: str, q: int, split_seed: int,
                       quotas: tuple[int, int, int], max_draws: int):
    """Exact message quotas with immutable digest ownership and bounded construction.

    Surplus messages stay unused, never reassigned. All numerical choices are
    explicit inputs; this function does not choose a confirmatory specification.
    """
    from .baselines import _sample_message
    if len(quotas) != 3 or min(quotas) < 1 or sum(quotas) != spec.size or max_draws < spec.size:
        raise ValueError("invalid quotas or construction budget")
    algorithm = normalize_algorithm(algorithm)
    digest_prefix_hex(hashlib.new(algorithm).digest(), q)
    rng, split_rng = random.Random(spec.seed), random.Random(split_seed)
    names = ("train", "validation", "test")
    partitions = {name: [] for name in names}
    owner, seen = {}, set()
    duplicates = unused = 0
    for draw in range(1, max_draws + 1):
        message = _sample_message(rng, spec.distribution, min_length=spec.min_length,
                                  max_length=spec.max_length, length=None)
        if message in seen:
            duplicates += 1
            continue
        seen.add(message)
        record = DigestRecord(draw - 1, spec.distribution, message, algorithm, q,
                              hashlib.new(algorithm, message).digest())
        if record.prefix not in owner:
            owner[record.prefix] = split_rng.choices(names, weights=quotas)[0]
        split_name = owner[record.prefix]
        if len(partitions[split_name]) < quotas[names.index(split_name)]:
            partitions[split_name].append(record)
        else:
            unused += 1
        if all(len(partitions[name]) == quota for name, quota in zip(names, quotas)):
            return {name: tuple(values) for name, values in partitions.items()}, {
                "draw_count": draw, "duplicate_count": duplicates, "unused_count": unused,
                "digest_ownership": owner, "max_draws": max_draws, "quotas": quotas,
            }
    raise RuntimeError("DATASET_CONSTRUCTION_FAILED: immutable group ownership could not fill quotas within max_draws")


__all__ = [
    "DigestRecord",
    "SourceSpec",
    "build_digest_records",
    "digest_prefix_hex",
    "generate_source_messages",
    "hash_caption",
    "normalize_algorithm",
    "select_digest_representatives",
    "split_validation_report",
    "split_digest_groups",
    "write_split",
]
