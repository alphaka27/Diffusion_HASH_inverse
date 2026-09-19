"""Same canonical (algorithm, q, exact prefix bits[, length]) for every model."""
import torch

from .dataset import normalize_algorithm

CONDITION_VERSION = "algorithm-q-prefix-msb-v1"


def digest_condition(algorithm: str, q: int, digest: bytes, *, length: int | None = None):
    algorithm = normalize_algorithm(algorithm)
    width = 128 if algorithm == "md5" else 256
    if len(digest) * 8 != width or not 1 <= q <= width:
        raise ValueError("digest width or q does not match algorithm")
    # Fixed dimensionality; trailing digest bits are zero, never leaked.
    values = [float(algorithm == "md5"), float(algorithm == "sha256"), q / 256]
    values += [(digest[index // 8] >> (7 - index % 8)) & 1 for index in range(q)]
    values += [0] * (256 - q)
    if length is not None:
        if not isinstance(length, int) or not 4 <= length <= 255:
            raise ValueError("known payload length must be in the BGV-supported domain")
        values.append(length / 255)
    return torch.tensor(values, dtype=torch.float32)


def shuffled_donors(records, *, seed: int, same_length: bool):
    """Deterministic digest derangement; reject impossible strata, never leak."""
    import random
    from collections import defaultdict
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[len(record.message) if same_length else None].append(index)
    donors = list(range(len(records)))
    rng = random.Random(seed)
    for indices in groups.values():
        by_digest = defaultdict(list)
        for index in indices:
            by_digest[records[index].prefix].append(index)
        keys = list(by_digest)
        rng.shuffle(keys)
        ordered = [index for key in keys for index in by_digest[key]]
        maximum = max(map(len, by_digest.values()))
        if maximum * 2 > len(ordered):
            raise ValueError("digest derangement impossible for a stratum")
        for position, index in enumerate(ordered):
            donors[index] = ordered[(position + maximum) % len(ordered)]
    if any(records[i].prefix == records[d].prefix for i, d in enumerate(donors)):
        raise RuntimeError("digest derangement invariant failed")
    return donors
