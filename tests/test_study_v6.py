"""V6 Phase 1 계약 검사. 조건부 데이터는 synthetic 또는 validation fixture만 쓴다."""
import ast
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from dhi_v5 import data as v5_data
from dhi_v6 import MASTER_SEED, PROTOCOL, data
from dhi_v6.protocol import (atomic_json, canonical, environment, file_hash, read_json,
                             registration, sealed_json, source_manifest)


def test_independence_and_registration(tmp_path):
    root = Path(__file__).parents[1]
    package = root / "src/dhi_v6"
    forbidden = {"dhi_v5", "diffusion_hash_inv", "torch"}
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in forbidden
            elif isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
    plan = root / "examples/v6-protocol.json"
    registered = registration()
    assert registered == json.loads(plan.read_text())
    assert canonical(registered) + b"\n" == plan.read_bytes()
    assert file_hash(plan) == plan.with_suffix(".json.sha256").read_text().strip()
    assert PROTOCOL == "dhi-v6-20260929" and MASTER_SEED == 2026092907
    assert registered["sources"] == data.SOURCES and registered["tokens"] == data.TOKENS
    assert registered["windows"] == data.WINDOWS
    assert registered["hash_gate"]["rungs"] == [*data.LADDER, 64]
    registered["training"]["batch"] = 1
    assert registration()["training"]["batch"] == 256
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert config["project"]["scripts"]["hash-inverse-v6"] == "dhi_v6.study:main"
    assert "metal: requires MLX Metal" in config["tool"]["pytest"]["ini_options"]["markers"]
    assert source_manifest() == {p.name: file_hash(p) for p in package.glob("*.py")}
    assert all(environment()[key] for key in ("python", "platform", "machine", "packages"))
    assert environment()["packages"]["numpy"] == np.__version__

    path = tmp_path / "nested/fixture.json"
    value = {"z": [1, True, None], "a": "fixture"}
    atomic_json(path, value)
    assert path.read_bytes() == canonical(value) + b"\n"
    assert not path.with_suffix(".json.tmp").exists()
    sealed_json(path, value)
    sealed_json(path, value)
    assert read_json(path) == value
    with pytest.raises(ValueError, match="Refusing"):
        sealed_json(path, {"changed": True})
    atomic_json(path, {"changed": True})
    with pytest.raises(ValueError, match="modified"):
        read_json(path)
    with pytest.raises(ValueError):
        canonical({"invalid": float("nan")})


def test_hash_windows_sources():
    for src in ("P", "R"):
        payload, lengths = data.source(data.rng("hash-gate", src), 1000, src)
        raw = data.messages(payload, lengths)
        assert all(data.valid(message, src) for message in raw)
        assert np.all(payload[np.arange(31) >= lengths[:, None]] == 0)
        for rung in (*data.LADDER, 64):
            vector = data.digest_batch(payload, lengths, rung)
            reference = [data.digest_reference(message, rung) for message in raw]
            assert vector.tobytes() == b"".join(reference)
            assert np.array_equal(vector, v5_data.digest_batch(payload, lengths, rung))
            if rung == 64:
                assert reference == [hashlib.md5(message).digest() for message in raw]
            for window, shift in data.WINDOWS.items():
                expected = [(int.from_bytes(d, "big") >> shift) & 4095 for d in reference]
                assert expected == [data.window_value(d, window) for d in reference]
                assert np.array_equal(data.hash_batch(payload, lengths, rung, window), expected)
                assert data.hash_one(raw[0], rung, window) == expected[0]
        changed, changed_lengths = data.source(data.rng("r4-suffix", src), 1000, src)
        changed[:, :4] = payload[:, :4]
        assert np.array_equal(data.hash_batch(changed, changed_lengths, 4, "W1"),
                              data.hash_batch(payload, lengths, 4, "W1"))
    for message in (b"", b"a", b"abc", b"message digest", b"abcdefghijklmnopqrstuvwxyz", b"x" * 55):
        payload = np.zeros((1, 55), dtype=np.uint8)
        payload[0, :len(message)] = np.frombuffer(message, dtype=np.uint8)
        digest = hashlib.md5(message).digest()
        assert data.digest_reference(message) == digest
        assert data.digest_batch(payload, np.array([len(message)])).tobytes() == digest
    with pytest.raises(ValueError, match="forbidden"):
        data.split("W2", 64)
    for task in ("md5", "synthetic"):
        with pytest.raises(ValueError, match="forbidden"):
            data.fresh_batch(("A-Q", "P", 0), 0, 8, [0], task=task, window="W2")
    with pytest.raises(ValueError):
        data.hash_batch(payload, np.array([55]), window="unknown")
    for rung in (0, 65, 4.5):
        with pytest.raises(ValueError):
            data.digest_reference(b"test", rung)
        with pytest.raises(ValueError):
            data.digest_batch(payload, np.array([55]), rung)
    with pytest.raises(ValueError):
        data.digest_batch(np.zeros((1, 3), dtype=np.uint8), np.array([4]))


def test_token_codec_p_r():
    for src, spec in data.SOURCES.items():
        ids = data.TOKENS[src]
        states, low, high = spec["states"], spec["byte_min"], spec["byte_max"]
        assert list(ids.values()) == [states, states + 1, states + 2, states + 3]
        symbols = np.arange(low, high + 1, dtype=np.uint8)
        for length in range(4, 32):
            payload = np.tile(symbols[:, None], (1, 31))
            lengths = np.full(states, length, dtype=np.int32)
            tokens = data.encode(payload, lengths, src)
            assert np.all(tokens[:, length] == ids["eos"])
            assert np.all(tokens[:, length + 1:] == ids["pad"])
            assert data.decode(tokens, src) == data.messages(payload, lengths)
            if src == "P":
                assert np.array_equal(tokens, v5_data.encode(payload, lengths))
            for invalid_length in (3, 32):
                assert not data.valid(bytes([low]) * invalid_length, src)
        good = data.encode(np.full((1, 31), low, dtype=np.int32), np.array([4]), src)
        for position, value in ((0, -1), (0, ids["mask"]), (0, ids["pad"]),
                                (0, ids["eos"]), (4, ids["pad"]), (5, 0)):
            bad = good.copy()
            bad[0, position] = value
            assert data.decode(bad, src) == [None]
        early = np.full((1, 32), ids["pad"], dtype=np.int32)
        early[0, :3], early[0, 3] = 0, ids["eos"]
        assert data.decode(early, src) == [None]
        assert data.decode(good[:, :31], src) == [None]
        assert data.decode(good.astype(float), src) == [None]
        assert not data.valid(None, src)
        with pytest.raises(ValueError):
            data.decode(good[0], src)
        for payload, lengths in ((np.full((1, 31), high + 1), [4]),
                                 (np.full((1, 31), low - 1), [4]),
                                 (np.full((1, 30), low), [4]),
                                 (np.full((1, 31), low), [3]),
                                 (np.full((1, 31), low), [32]),
                                 (np.full((1, 31), low), [4.5]),
                                 (np.full((1, 31), low + .5), [4])):
            with pytest.raises(ValueError):
                data.encode(payload, lengths, src)
    assert not data.valid(b"abc\x00", "P") and data.valid(b"abc\x00", "R")
    assert data.condition_bits([0, 4095, 2048]).tolist() == [[0] * 12, [1] * 12, [1] + [0] * 11]
    for values in ([-1], [4096], [1.5], [[0]]):
        with pytest.raises(ValueError):
            data.condition_bits(values)


def test_shared_streams_and_controls(monkeypatch):
    ns = ("A-Q", "P", 0)
    expected = hashlib.sha256(canonical([PROTOCOL, MASTER_SEED, *ns])).hexdigest()
    assert data.identity(*ns) == expected and data.seed(*ns) == int(expected[:16], 16)
    assert data.identity(*ns) != v5_data.identity(*ns)
    indices = np.array([0, 1, 99, 100, 2**32, 2**64 - 1], dtype=np.uint64)
    keys = data.key_words(ns, indices)
    for index, key in zip(indices, keys, strict=True):
        z = (int(index) + data.seed("candidate", *ns) + 0x9E3779B97F4A7C15) & (2**64 - 1)
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
        z ^= z >> 31
        assert key.tolist() == [z >> 32, z & 0xFFFFFFFF]
    for draw in (0, 1, 7, 64):
        expected = keys ^ np.array([(0x9E3779B9 * (draw + 1)) & 0xFFFFFFFF,
                                    (0x85EBCA6B * (draw + 1)) & 0xFFFFFFFF], dtype=np.uint32)
        assert np.array_equal(data.draw_key(keys, draw), expected)

    synthetic = data.synthetic_split()
    assert {k: len(v) for k, v in synthetic.items()} == {
        "test": 1024, "acceptance": 512, "validation": 256, "train": 2816}
    assert set(synthetic["acceptance"]) <= set(synthetic["test"])
    for role in ("test", "acceptance", "validation", "train"):
        assert set(synthetic[role]) == {g ^ 4095 for g in synthetic[role]}
    assert len(set(synthetic["test"] + synthetic["validation"] + synthetic["train"])) == 4096

    for window, rung in (("W1", 64), ("W1", 4), ("W3", 64), ("W4", 64)):
        order = data.rng("ownership", window, rung).permutation(4096).tolist()
        excluded = order[:16]
        groups = data.split(window, rung, excluded)
        assert groups == data.split(window, rung, excluded)
        assert groups["test"] == order[16:1040]
        remaining = [g for g in order if g not in set(groups["test"])]
        assert groups["validation"] == remaining[:256] and groups["train"] == remaining[256:]
        assert len(groups["train"]) == 2816
        for src in ("P", "R"):
            # Validation fixtures never create or evaluate protected test candidates.
            ns = ("A-impl", src, 0)
            options = {"window": window, "rung": rung}
            a = data.fresh_batch(ns, 7, 32, groups["validation"], **options)
            b = data.fresh_batch(ns, 7, 32, groups["validation"], **options)
            assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
            assert np.isin(a[2], groups["validation"]).all()
            assert not np.isin(a[2], groups["test"]).any()
            assert np.array_equal(a[2], data.hash_batch(a[0], a[1], rung, window))
            assert a[3] >= 32
    for src in ("P", "R"):
        ns = ("A-Q", src, 0)
        shared = data.fresh_batch(ns, 7, 256, synthetic["train"], task="synthetic")
        payload, lengths, labels, calls = shared
        assert calls == 0 and np.isin(labels, synthetic["train"]).all()
        assert [data.synthetic_label(x, src) for x in data.messages(payload, lengths)] == labels.tolist()
        generator = data.rng("fresh", *ns, "synthetic", "W1", 64, 7)
        expected_payload, expected_lengths = data.source(generator, 256, src)
        assert np.array_equal(lengths, expected_lengths)
        assert np.array_equal(payload[:, 3:], expected_payload[:, 3:])
        assert np.array_equal(labels, generator.choice(synthetic["train"], 256))
        permutation = data.shuffle_permutation(ns, 7, 256)
        assert np.array_equal(permutation, data.rng("shuffle", *ns, 7).permutation(256))
        assert np.array_equal(np.sort(permutation), np.arange(256))
        for pipeline in registration()["pipelines"].values():
            if pipeline["source"] != src:
                continue
            for method in ("Main", "Shuffled"):
                batch = data.fresh_batch(ns, 7, 256, synthetic["train"], task="synthetic")
                assert all(np.array_equal(x, y) for x, y in zip(shared, batch, strict=True))
        other = data.fresh_batch(ns, 8, 256, synthetic["train"], task="synthetic")
        assert not np.array_equal(payload, other[0])
        clp = data.fresh_batch(("clp", "A-Q", "fixture", src, 0), 0, 8,
                               synthetic["validation"], task="synthetic")
        assert np.isin(clp[2], synthetic["validation"]).all()
        prior_ns = ("A-prof", src, "W1", 64, "Random", 0)
        prior = data.prior_candidates(prior_ns, indices, src)
        singles = [data.prior_candidates(prior_ns, [i], src) for i in indices]
        assert all(np.array_equal(prior[k], np.concatenate([s[k] for s in singles])) for k in (0, 1))
        assert all(data.valid(m, src) for m in data.messages(*prior))
        assert np.all(prior[0][np.arange(31) >= prior[1][:, None]] == 0)
    assert data.synthetic_label(b"ABC!", "P") == 0xABC
    assert data.synthetic_label(bytes([10, 11, 12, 255]), "R") == 0xABC
    for message, src in ((b"abc!", "P"), (b"AB", "P"), (b"\x00\x01", "R"),
                         (b"\x10\x00\x00!", "R"), (b"", "P")):
        assert data.synthetic_label(message, src) == -1
    for size in (2, 100, 1024):
        donors = data.derangement(("A-Q", "P-DISC", "W1", 64, 0), size)
        assert np.all(donors != np.arange(size)) and np.array_equal(np.sort(donors), np.arange(size))
    with pytest.raises(ValueError):
        data.derangement(ns, 1)
    with pytest.raises(ValueError):
        data.split("W3", 64, range(3073))
    with pytest.raises(ValueError):
        data.split("W3", 64, [-1])
    with pytest.raises(ValueError):
        data.fresh_batch(ns, 0, 8, [], task="synthetic")
    with pytest.raises(ValueError):
        data.fresh_batch(("A-Q", "P-DISC", 0), 0, 8, [0], task="synthetic")

    # Force the rare 32-bit rejection boundary; R payload draws must accept it.
    calls = []
    def boundary_keys(namespace, indices):
        position, draw = namespace[-2:]
        calls.append((position, draw))
        word = 0xFFFFFFFF if draw == 0 else 0
        return np.full((len(indices), 2), word, dtype=np.uint32)
    with monkeypatch.context() as patch:
        patch.setattr(data, "key_words", boundary_keys)
        for src, byte in (("P", 33), ("R", 255)):
            calls.clear()
            payload, lengths = data.prior_candidates(("boundary", src), [0], src)
            assert lengths.tolist() == [4] and payload[0, :4].tolist() == [byte] * 4
            assert calls.count((0, 1)) == 1
            assert sum(draw == 1 for _, draw in calls) == (32 if src == "P" else 1)


def test_image_encoders_match_v31():
    import torch
    from dhi_v6 import codecs
    from diffusion_hash_inv.encoding.bgv import BGVEncoder
    from diffusion_hash_inv.encoding.cgge import CGGEEncoder
    for src, representation, encoder in (("P", "bgv", BGVEncoder()), ("R", "bgv", BGVEncoder()),
                                         ("P", "cgge", CGGEEncoder())):
        payload, lengths = data.source(data.rng("codec-parity", src), 1000, src)
        lengths[:2] = [4, 31]
        payload[0, 4:] = 0
        payload[1] = data.SOURCES[src]["byte_min"]
        images = codecs.encode(payload, lengths, representation, src)
        expected = torch.stack([encoder.encode(m) for m in data.messages(payload, lengths)]).numpy() * 2 - 1
        assert images.dtype == np.float32 and np.array_equal(images, expected)
        fixed, mask = codecs.structure(lengths, representation)
        assert np.array_equal(images[~mask], fixed[~mask])
        assert np.all(mask[:, 1] == 0)
        assert np.array_equal(mask.sum(axis=(1, 2, 3)), lengths * (128 if representation == "bgv" else 64))


def test_prototype_and_strict_decoders():
    import time
    import torch
    from dhi_v6 import codecs
    from diffusion_hash_inv.encoding.bgv import BGVDecoder
    from diffusion_hash_inv.encoding.cgge import CGGEDecoder, glyph_table_checksum
    assert codecs.glyph_table_checksum() == glyph_table_checksum()
    assert codecs.glyph_table_checksum().startswith("6ef6d0bf") and codecs.glyph_table_checksum().endswith("ed50a")
    for src, representation, decoder in (("P", "bgv", BGVDecoder()), ("R", "bgv", BGVDecoder()),
                                         ("P", "cgge", CGGEDecoder())):
        spec = data.SOURCES[src]
        payload = np.tile(np.arange(spec["byte_min"], spec["byte_max"] + 1, dtype=np.uint8)[:, None], (1, 31))
        for length in range(4, 32):
            lengths = np.full(len(payload), length, dtype=np.int32)
            images = codecs.encode(payload, lengths, representation, src)
            decoded, margin, strict = codecs.decode(images, lengths, representation, src)
            assert decoded == data.messages(payload, lengths)
            assert np.all(margin == 0) and strict.all()
            assert codecs.strict_decode(images, lengths, representation, src) == decoded
        generator = data.rng("strict-fixture", src, representation)
        payload, lengths = data.source(generator, 200, src)
        images = codecs.encode(payload, lengths, representation, src)
        fixed, mask = codecs.structure(lengths, representation)
        noise = generator.normal(0, .6, images.shape).astype(np.float32)
        images = np.where(mask, np.clip(images + noise, -1, 1), fixed)
        images[100:] = np.where(mask[100:], generator.uniform(-1, 1, images[100:].shape), fixed[100:])
        expected = [decoder.decode(torch.from_numpy(row.copy()), normalized=True).message for row in images]
        expected = [m if data.valid(m, src) else None for m in expected]
        assert codecs.strict_decode(images, lengths, representation, src) == expected
        decoded, _, strict = codecs.decode(images, lengths, representation, src)
        assert all(data.valid(m, src) for m in decoded)
        assert strict.tolist() == [m is not None for m in expected]
        payload, lengths = data.source(data.rng("codec-benchmark", src), 2048, src)
        times = []
        for _ in range(4):
            start = time.perf_counter()
            codecs.encode(payload[:256], lengths[:256], representation, src)
            times.append(time.perf_counter() - start)
        images = codecs.encode(payload, lengths, representation, src)
        start = time.perf_counter()
        codecs.decode(images, lengths, representation, src)
        elapsed = time.perf_counter() - start
        print(f"{src}-{representation}: encode256_ms={np.median(times[1:])*1000:.3f}, decode2048_s={elapsed:.4f}")
    for src, expected in (("P", b"!!!!"), ("R", bytes(4))):
        fixed, mask = codecs.structure(np.array([4]), "bgv")
        image = np.where(mask, 0., fixed).astype(np.float32)
        decoded, margins, _ = codecs.decode(image, [4], "bgv", src)
        assert decoded == [expected] and margins.tolist() == [2.]
    payload = np.full((1, 31), ord("I"), dtype=np.uint8)
    a = codecs.encode(payload, [4], "cgge", "P")
    b = codecs.encode(np.full_like(payload, ord("l")), [4], "cgge", "P")
    assert codecs.decode((a + b) / 2, [4], "cgge", "P")[0] == [b"IIII"]
    with pytest.raises(ValueError):
        codecs.encode(payload, [4], "cgge", "R")
    with pytest.raises(FloatingPointError):
        codecs.decode(a * np.nan, [4], "cgge", "P")
