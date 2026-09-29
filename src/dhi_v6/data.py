"""두 source의 결정적 데이터 계층. dhi_v5/data.py를 복사해 V6 규칙으로 확장했다."""
import hashlib
import math
import struct

import numpy as np

from . import MASTER_SEED, PROTOCOL
from .protocol import canonical

WIDTH = 32
SOURCES = {"P": {"byte_min": 33, "byte_max": 126, "states": 94},
           "R": {"byte_min": 0, "byte_max": 255, "states": 256}}
TOKENS = {src: {"pad": spec["states"], "eos": spec["states"] + 1,
                "mask": spec["states"] + 2, "vocab": spec["states"] + 3}
          for src, spec in SOURCES.items()}
LADDER = (4, 5, 6, 7, 8, 10, 12, 16, 32)
WINDOWS = {"W1": 116, "W2": 0, "W3": 52, "W4": 84}
IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)
SHIFTS = (7, 12, 17, 22) * 4 + (5, 9, 14, 20) * 4 + (4, 11, 16, 23) * 4 + (6, 10, 15, 21) * 4
CONSTANTS = tuple(int(abs(math.sin(i + 1)) * 2**32) for i in range(64))
HEX = np.frombuffer(b"0123456789ABCDEF", dtype=np.uint8)


def identity(*parts):
    return hashlib.sha256(canonical([PROTOCOL, MASTER_SEED, *parts])).hexdigest()


def seed(*parts):
    return int(identity(*parts)[:16], 16)


def rng(*parts):
    return np.random.default_rng(seed(*parts))


def source(generator, count, src):
    spec = SOURCES[src]
    lengths = generator.integers(4, 32, count, dtype=np.int32)
    payload = generator.integers(spec["byte_min"], spec["byte_max"] + 1, (count, 31), dtype=np.uint8)
    payload[np.arange(31)[None, :] >= lengths[:, None]] = 0
    return payload, lengths


def messages(payload, lengths):
    return [row[:int(n)].tobytes() for row, n in zip(payload, lengths, strict=True)]


def valid(message, src):
    spec = SOURCES[src]
    return (isinstance(message, bytes) and 4 <= len(message) <= 31
            and all(spec["byte_min"] <= b <= spec["byte_max"] for b in message))


def encode(payload, lengths, src):
    spec, ids = SOURCES[src], TOKENS[src]
    payload, lengths = np.asarray(payload), np.asarray(lengths)
    if (lengths.ndim != 1 or payload.shape != (len(lengths), 31)
            or not np.issubdtype(lengths.dtype, np.integer)
            or not np.issubdtype(payload.dtype, np.integer)
            or np.any((lengths < 4) | (lengths > 31))):
        raise ValueError("Integer payload must have lengths 4..31 and width 31")
    active = np.arange(31)[None, :] < lengths[:, None]
    if np.any(((payload < spec["byte_min"]) | (payload > spec["byte_max"])) & active):
        raise ValueError("Payload is outside the source alphabet")
    tokens = np.full((len(lengths), WIDTH), ids["pad"], dtype=np.int32)
    tokens[:, :31] = np.where(active, payload.astype(np.int32) - spec["byte_min"], ids["pad"])
    tokens[np.arange(len(lengths)), lengths] = ids["eos"]
    return tokens


def decode(tokens, src):
    spec, ids = SOURCES[src], TOKENS[src]
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("Tokens must be a batch of rows")
    result = []
    for row in tokens:
        eos = np.flatnonzero(row == ids["eos"])
        if row.shape != (WIDTH,) or len(eos) != 1 or not np.issubdtype(row.dtype, np.integer):
            result.append(None)
            continue
        n = int(eos[0])
        ok = (4 <= n <= 31 and np.all((row[:n] >= 0) & (row[:n] < spec["states"]))
              and np.all(row[n + 1:] == ids["pad"]))
        result.append(bytes((row[:n] + spec["byte_min"]).tolist()) if ok else None)
    return result


def condition_bits(values):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values > 4095)):
        raise ValueError("Conditions must be integer 12-bit values")
    return ((values[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.float32)


def digest_reference(message, steps=64):
    """RFC 1321 scalar reference, independent of the vectorized implementation."""
    if (not isinstance(message, bytes) or len(message) > 55
            or not isinstance(steps, (int, np.integer)) or not 1 <= steps <= 64):
        raise ValueError("One-block MD5 requires <=55 bytes and 1..64 steps")
    block = message + b"\x80" + bytes(55 - len(message)) + struct.pack("<Q", 8 * len(message))
    words = struct.unpack("<16I", block)
    a, b, c, d = IV
    for i in range(steps):
        if i < 16:
            f, g = (b & c) | (~b & d), i
        elif i < 32:
            f, g = (d & b) | (~d & c), (5 * i + 1) % 16
        elif i < 48:
            f, g = b ^ c ^ d, (3 * i + 5) % 16
        else:
            f, g = c ^ (b | ~d), (7 * i) % 16
        v = (a + f + CONSTANTS[i] + words[g]) & 0xFFFFFFFF
        v = ((v << SHIFTS[i]) | (v >> (32 - SHIFTS[i]))) & 0xFFFFFFFF
        a, b, c, d = d, (b + v) & 0xFFFFFFFF, b, c
    return struct.pack("<4I", *((x + y) & 0xFFFFFFFF for x, y in zip((a, b, c, d), IV)))


def digest_batch(payload, lengths, steps=64):
    payload, lengths = np.asarray(payload), np.asarray(lengths)
    if (payload.ndim != 2 or lengths.ndim != 1 or len(payload) != len(lengths)
            or not np.issubdtype(payload.dtype, np.integer)
            or not np.issubdtype(lengths.dtype, np.integer)
            or np.any((payload < 0) | (payload > 255))
            or np.any((lengths < 0) | (lengths > 55))
            or not isinstance(steps, (int, np.integer)) or not 1 <= steps <= 64):
        raise ValueError("Invalid single-block inputs")
    if np.any(lengths > payload.shape[1]):
        raise ValueError("Payload is shorter than its declared length")
    block = np.zeros((len(payload), 64), dtype=np.uint8)
    width = min(payload.shape[1], 55)
    block[:, :width] = np.where(np.arange(width) < lengths[:, None], payload[:, :width], 0)
    block[np.arange(len(payload)), lengths] = 128
    block[:, 56:64] = (lengths.astype("<u8") * 8).view(np.uint8).reshape(-1, 8)
    words = block.view("<u4")
    a, b, c, d = [np.full(len(payload), n, dtype=np.uint32) for n in IV]
    for i in range(steps):
        if i < 16:
            f, g = (b & c) | (~b & d), i
        elif i < 32:
            f, g = (d & b) | (~d & c), (5 * i + 1) % 16
        elif i < 48:
            f, g = b ^ c ^ d, (3 * i + 5) % 16
        else:
            f, g = c ^ (b | ~d), (7 * i) % 16
        v = a + f + np.uint32(CONSTANTS[i]) + words[:, g]
        v = (v << SHIFTS[i]) | (v >> (32 - SHIFTS[i]))
        a, b, c, d = d, b + v, b, c
    state = np.stack([x + np.uint32(y) for x, y in zip((a, b, c, d), IV)], axis=1).astype("<u4")
    return state.view(np.uint8).reshape(-1, 16)


def window_value(digest, window):
    return (int.from_bytes(digest, "big") >> WINDOWS[window]) & 4095


def hash_batch(payload, lengths, steps=64, window="W1"):
    digest = digest_batch(payload, lengths, steps).astype(np.uint16)
    if window == "W1":
        return (digest[:, 0] << 4) | (digest[:, 1] >> 4)
    if window == "W2":
        return ((digest[:, 14] & 15) << 8) | digest[:, 15]
    if window == "W3":
        return (digest[:, 8] << 4) | (digest[:, 9] >> 4)
    if window == "W4":
        return (digest[:, 4] << 4) | (digest[:, 5] >> 4)
    raise ValueError("Unknown digest window")


def synthetic_label(message, src):
    if src not in SOURCES:
        raise ValueError("Unknown source")
    if not isinstance(message, bytes) or len(message) < 3:
        return -1
    if src == "P":
        return int(message[:3], 16) if all(b in HEX for b in message[:3]) else -1
    return (message[0] << 8) | (message[1] << 4) | message[2] if all(b < 16 for b in message[:3]) else -1


def hash_one(message, steps=64, window="W1", task="md5", src="P"):
    if task == "synthetic":
        return synthetic_label(message, src)
    if task != "md5":
        raise ValueError("Unknown task")
    return window_value(hashlib.md5(message).digest() if steps == 64 else digest_reference(message, steps), window)


def split(window, rung, excluded=()):
    if window not in WINDOWS or window == "W2":
        raise ValueError("Unknown or forbidden ownership window")
    if not isinstance(rung, (int, np.integer)) or not 1 <= rung <= 64:
        raise ValueError("MD5 requires 1..64 steps")
    excluded = set(excluded)
    if any(type(x) is not int or not 0 <= x < 4096 for x in excluded):
        raise ValueError("Invalid exposure groups")
    order = rng("ownership", window, rung).permutation(4096).tolist()
    clean = [x for x in order if x not in excluded]
    if len(clean) < 1024:
        raise ValueError("Fewer than 1,024 unexposed groups")
    test = clean[:1024]
    test_set = set(test)
    remaining = [x for x in order if x not in test_set]
    return {"test": test, "validation": remaining[:256], "train": remaining[256:]}


def synthetic_split():
    # Complement pairs stay together: flipped acceptance never enters training/dev.
    pairs = rng("synthetic-ownership").permutation(2048)
    expand = lambda a: np.concatenate((a, a ^ 4095)).astype(int).tolist()
    return {"test": expand(pairs[:512]), "acceptance": expand(pairs[:256]),
            "validation": expand(pairs[512:640]), "train": expand(pairs[640:])}


def fresh_batch(namespace, update, count, groups, *, task="md5", window="W1", rung=64):
    """Stateless per-update rejection stream. Main/Shuffled pass the same namespace."""
    if window not in WINDOWS or window == "W2":
        raise ValueError("Unknown or forbidden stream window")
    if task not in ("md5", "synthetic"):
        raise ValueError("Unknown task")
    if not (len(namespace) == 3 or (len(namespace) == 5 and namespace[0] == "clp")):
        raise ValueError("Stream namespace must be (stage, source, seed) or (clp, stage, purpose, source, seed)")
    src = namespace[-2]
    if src not in SOURCES:
        raise ValueError("Unknown source")
    allowed = np.zeros(4096, dtype=bool)
    group_array = np.asarray(groups)
    condition_bits(group_array)
    if (not len(group_array) or not isinstance(count, (int, np.integer)) or count <= 0
            or not isinstance(update, (int, np.integer)) or update < 0):
        raise ValueError("Empty ownership or invalid batch request")
    allowed[group_array] = True
    generator = rng("fresh", *namespace, task, window, rung, update)
    if task == "synthetic":
        payload, lengths = source(generator, count, src)
        labels = generator.choice(group_array, count).astype(np.int32)
        nibbles = (labels[:, None] >> np.array([8, 4, 0])) & 15
        payload[:, :3] = HEX[nibbles] if src == "P" else nibbles
        return payload, lengths, labels, 0
    chunks, size, calls = [], 0, 0
    for _ in range(10000):
        n = max(256, math.ceil((count - size) * 4096 / len(group_array) * 1.1))
        payload, lengths = source(generator, n, src)
        labels = hash_batch(payload, lengths, rung, window)
        keep = allowed[labels]
        chunks.append((payload[keep], lengths[keep], labels[keep]))
        size += int(keep.sum())
        calls += n
        if size >= count:
            arrays = tuple(np.concatenate([c[i] for c in chunks])[:count] for i in range(3))
            return *arrays, calls
    raise RuntimeError("Rejection stream exhausted its deterministic draw cap")


def shuffle_permutation(namespace, update, size):
    return rng("shuffle", *namespace, update).permutation(size)


def derangement(namespace, size):
    if size < 2:
        raise ValueError("MC needs at least two trials")
    order = rng("mc-derangement", *namespace).permutation(size)
    donor = np.empty(size, dtype=np.int32)
    donor[order] = np.roll(order, 1)
    return donor


def key_words(namespace, indices):
    indices = np.asarray(indices, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = indices + np.uint64(seed("candidate", *namespace)) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
        z ^= z >> 31
    return np.stack((z >> 32, z & 0xFFFFFFFF), axis=1).astype(np.uint32)


def draw_key(key, draw):
    return np.asarray(key, dtype=np.uint32) ^ np.array(
        [(0x9E3779B9 * (draw + 1)) & 0xFFFFFFFF,
         (0x85EBCA6B * (draw + 1)) & 0xFFFFFFFF], dtype=np.uint32)


def prior_candidates(namespace, indices, src):
    """Vectorized, batch-invariant source prior; rejection avoids modulo bias."""
    spec = SOURCES[src]
    indices = np.asarray(indices, dtype=np.uint64)
    values = np.zeros((len(indices), 32), dtype=np.int32)
    for position in range(32):
        bound, low = (28, 4) if position == 0 else (spec["states"], spec["byte_min"])
        pending = np.arange(len(indices))
        draw = 0
        while len(pending):
            words = key_words((*namespace, "prior", position, draw), indices[pending])[:, 1]
            keep = words.astype(np.uint64) < (2**32 // bound) * bound
            values[pending[keep], position] = words[keep] % bound + low
            pending = pending[~keep]
            draw += 1
    lengths, payload = values[:, 0], values[:, 1:].astype(np.uint8)
    payload[np.arange(31) >= lengths[:, None]] = 0
    return payload, lengths
