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


def run_metal(code, *args):
    import os
    import subprocess
    import sys
    available = subprocess.run([sys.executable, "-c", "import mlx.core as mx; a=mx.ones((64,64)); mx.eval(a@a); assert mx.default_device()==mx.gpu"], capture_output=True, text=True)
    if available.returncode:
        if os.environ.get("DHI_V6_REQUIRE_METAL") == "1":
            pytest.fail("Required Metal unavailable: " + available.stderr)
        pytest.skip("Metal unavailable")
    result = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")


@pytest.mark.metal
def test_parameter_counts():
    run_metal('''
from dhi_v6.models import make_model, parameter_count
expected = {"P-DISC":508668,"R-DISC":1252572,"P-G-BGV":910638,"R-G-BGV":910638,"P-G-CGGE":513326}
for p, count in expected.items():
    assert parameter_count(make_model(p,"A-Q",0)) == count
assert parameter_count(make_model("P-DISC","S",0,"D1-T-L")) == 6415308
''')


@pytest.mark.metal
def test_d1s_parity_with_v5():
    run_metal('''
import numpy as np
from mlx.utils import tree_flatten
from dhi_v5 import models as old
from dhi_v6 import models, data
reference=old.make_model("D1-S",("v6-parity",))
model=models.make_model("P-DISC","A-Q",0)
model.load_weights(tree_flatten(reference.parameters()))
labels=np.array(data.synthetic_split()["acceptance"][:256])
keys=models.candidate_keys(("A-Q","P-DISC","normal",0),np.arange(256))
assert np.array_equal(old.sample(reference,labels,keys),models.sample_tokens(model,labels,keys))
''')


@pytest.mark.metal
def test_g3u_forward_parity_with_v31():
    run_metal('''
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from diffusion_hash_inv.mlx_models import ImageUNet, LengthGaussianDiffusion
from dhi_v6 import models, codecs, data
for p,rep in (("P-G-BGV","bgv"),("P-G-CGGE","cgge")):
    model=models.make_model(p,"A-Q",0)
    reference=ImageUNet(codecs.SHAPES[rep],32,coordinates=True,condition_output=True,factorized_length=True)
    reference.load_weights(tree_flatten(model.parameters()))
    lengths=mx.array(np.tile(np.arange(4,32),3)[:64].astype(np.int32))
    fixed,mask=models.structure(lengths,rep)
    oldfixed,oldmask=LengthGaussianDiffusion(1000,prediction_type="sample",loss_regions=rep).structure(lengths)
    cpu_fixed,cpu_mask=codecs.structure(np.asarray(lengths),rep)
    assert np.array_equal(np.asarray(fixed),np.asarray(oldfixed)) and np.array_equal(np.asarray(mask),np.asarray(oldmask))
    assert np.array_equal(np.asarray(fixed),cpu_fixed) and np.array_equal(np.asarray(mask),cpu_mask)
    x=mx.random.normal((64,*codecs.SHAPES[rep]),key=mx.random.key(10))
    t=mx.full((64,),.4)
    c=mx.concatenate((mx.array(data.condition_bits(np.arange(64))),lengths[:,None]/31),axis=1)
    assert np.array_equal(np.asarray(model(x,t,c)),np.asarray(reference(x,t,c)))
''')


@pytest.mark.metal
def test_samplers_reference_and_invariance():
    run_metal('''
import numpy as np
from dhi_v6 import models, data, codecs
labels=np.array(data.synthetic_split()["acceptance"][:256])
for p in ("P-DISC","R-DISC"):
    model=models.make_model(p,"A-Q",0)
    keys=models.candidate_keys(("A-Q",p,"normal",0),np.arange(256))
    reference=models.sample_tokens_reference(model,labels,keys)
    for batch in (64,128,256):
        actual=np.concatenate([models.sample_tokens(model,labels[i:i+batch],keys[i:i+batch]) for i in range(0,256,batch)])
        assert np.array_equal(actual,reference),(p,batch)
for p in ("P-G-BGV","P-G-CGGE","R-G-BGV"):
    model=models.make_model(p,"A-Q",0)
    keys=models.candidate_keys(("A-Q",p,"normal",0),np.arange(256))
    images,lengths=models.sample_images(model,labels,keys)
    for batch in (64,128):
        pieces=[models.sample_images(model,labels[i:i+batch],keys[i:i+batch]) for i in range(0,256,batch)]
        assert np.array_equal(images,np.concatenate([x[0] for x in pieces])),(p,batch)
        assert np.array_equal(lengths,np.concatenate([x[1] for x in pieces]))
    reference,rl=models.sample_images_reference(model,labels[:64],keys[:64])
    assert np.array_equal(lengths[:64],rl)
    error=np.max(np.abs(reference-images[:64]))
    assert error <= 1e-3,(p,error)
    a=codecs.decode(reference,rl,model.representation,model.src)[0]
    b=codecs.decode(images[:64],rl,model.representation,model.src)[0]
    assert sum(x==y for x,y in zip(a,b))>=63,(p,error)
model=models.make_model("P-DISC","S",0,"D1-T-L")
keys=models.candidate_keys(("A-Q","scale","normal",0),np.arange(8))
assert np.array_equal(models.sample_tokens(model,labels[:8],keys),models.sample_tokens_reference(model,labels[:8],keys))
''')


@pytest.mark.metal
def test_clp_antisymmetry():
    run_metal('''
import numpy as np
from dhi_v6 import models, data, codecs
from dhi_v6.protocol import registration
for p,spec in registration()["pipelines"].items():
    model=models.make_model(p,"A-Q",0)
    payload,lengths,labels,_=data.fresh_batch(("clp","A-Q","fixture",spec["source"],0),0,8,data.synthetic_split()["validation"],task="synthetic")
    clean=data.encode(payload,lengths,spec["source"]) if spec["representation"]=="tokens" else codecs.encode(payload,lengths,spec["representation"],spec["source"])
    keys=models.candidate_keys(("clp-corruption","A-Q","fixture",p,0),np.arange(4))
    d=models.clp_pairs(model,clean,lengths,labels,keys)
    opposite=models.clp_pairs(model,clean,lengths,labels.reshape(-1,2)[:,::-1].reshape(-1),keys)
    assert np.array_equal(d,-opposite),p
''')


@pytest.mark.metal
def test_training_resume_and_continuation(tmp_path):
    run_metal('''
import sys
from pathlib import Path
import numpy as np
from mlx.utils import tree_flatten
from dhi_v6 import runtime, data
from dhi_v6.protocol import read_json
root=Path(sys.argv[1])
def equal(a,b):
    for (ka,va),(kb,vb) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters()),strict=True):
        assert ka==kb and np.array_equal(np.asarray(va),np.asarray(vb)),ka
for p in ("P-DISC","R-DISC","P-G-BGV","P-G-CGGE"):
    kwargs=dict(pipeline=p,stage="A-Q",seed_id=0,groups=data.synthetic_split(),task="synthetic",batch_size=8,checkpoint_every=2,diagnostic_pairs=2)
    direct=runtime.train(root/p/"direct",updates=8,**kwargs)
    original=runtime.save_checkpoint
    def interrupted(folder,model,optimizer,update):
        original(folder,model,optimizer,update)
        if update==2: raise InterruptedError("checkpoint interruption")
    runtime.save_checkpoint=interrupted
    try:
        runtime.train(root/p/"resumed",updates=4,**kwargs)
        raise AssertionError("not interrupted")
    except InterruptedError: pass
    finally: runtime.save_checkpoint=original
    resumed=runtime.train(root/p/"resumed",updates=4,**kwargs)
    four=runtime.train(root/p/"four",updates=4,**kwargs)
    equal(resumed,four)
    continued=runtime.train(root/p/"continued",updates=8,resume_from=root/p/"four",**kwargs)
    equal(direct,continued)
    assert read_json(root/p/"resumed/attempt.json")["retries"]==1
    assert read_json(root/p/"continued/contract.json")["resume_from"]["update"]==4
# The same source shares W1 r=64 fixture digests across two actual pipelines.
groups=data.split("W1",64)
kwargs=dict(stage="A-prof",seed_id=0,groups=groups,task="md5",window="W1",updates=2,batch_size=8,checkpoint_every=2,diagnostic_pairs=2,stream_root=root/"streams")
for p in ("P-DISC","P-G-BGV"):
    runtime.train(root/"shared"/p,p,**kwargs)
a=runtime.verify_training(root/"shared/P-DISC")["digest_segments"]
b=runtime.verify_training(root/"shared/P-G-BGV")["digest_segments"]
assert a==b and len(a)==1
path=Path(next(iter(a)))
arr=np.load(path); arr[0]=np.void(bytes(16)); runtime.atomic_array(path,value=arr)
try:
    runtime.train(root/"shared/P-G-CGGE","P-G-CGGE",**kwargs)
    raise AssertionError("digest mismatch not caught")
except ValueError as e: assert "digest mismatch" in str(e)
''', tmp_path)


def test_ledger_commit_resume_tamper_regeneration(tmp_path):
    from dhi_v6 import runtime
    assert runtime.RECORD.itemsize == 36
    assert [runtime.RECORD.fields[k][1] for k in runtime.RECORD.names] == [0, 31, 32, 33, 35]
    targets = np.array(data.split("W1", 64)["validation"][:128])
    options = dict(stage="A-prof", source="P", method="Random", seed_id=0, batch=256)
    summary, result = runtime.evaluate_block(tmp_path / "direct", 1, targets, **options)
    class Stop:
        calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 12:
                raise InterruptedError("partial block")
    with pytest.raises(InterruptedError):
        runtime.evaluate_block(tmp_path / "resumed", 1, targets, budget=Stop(), **options)
    assert read_json(tmp_path / "resumed/block-1.commit.json")["committed_rows"] == 2048
    resumed, _ = runtime.evaluate_block(tmp_path / "resumed", 1, targets, **options)
    assert np.array_equal(summary, resumed)
    assert (tmp_path / "direct/block-1.bin").read_bytes() == (tmp_path / "resumed/block-1.bin").read_bytes()
    records = np.memmap(tmp_path / "resumed/block-1.bin", dtype=runtime.RECORD, mode="r+")
    records["flags"][0] ^= 2
    records.flush()
    with pytest.raises(ValueError, match="flags"):
        runtime.verify_ledger(tmp_path / "resumed/block-1.bin", targets, result["metadata"])
    records["flags"][0] ^= 2
    namespace = result["metadata"]["namespace"]
    all_indices = np.arange(len(records), dtype=np.uint64)
    chosen = all_indices[data.key_words(("regen-audit", namespace, 1), all_indices)[:, 1] % 100 == 0]
    index = int(chosen[0])
    records["payload"][index, 0] = 33 if records["payload"][index, 0] != 33 else 34
    records.flush()
    def generate(indices):
        p, n = data.prior_candidates(namespace, indices, "P")
        return data.messages(p, n), None, None
    with pytest.raises(ValueError, match="Regeneration"):
        runtime.regeneration_audit(tmp_path / "resumed/block-1.bin", namespace, 1, 0, len(records), generate)
    meta = {"source": "P", "task": "synthetic", "rung": 64, "window": "W1", "method": "Random"}
    ledger = runtime.Ledger(tmp_path / "fixture", 1, meta, [0, 0], batch=3, k=3)
    ledger.append(0, [b"000!", b"000!", None])
    ledger.append(3, [b"FFFF", b"000!", b"~~~~"])
    ledger.commit()
    ledger.close()
    trials, metrics = runtime.verify_ledger(ledger.path, [0, 0], meta, k=3)
    assert metrics["rows"] == 6 and metrics["hits"] == 3 and metrics["duplicates"] == 1
    assert trials["at100"].tolist() == [1, 1] and trials["first"].tolist() == [0, 1]
    training = runtime.message_digests([b"000!"])
    ledger = runtime.Ledger(tmp_path / "match", 1, meta, [0], batch=1, k=1, training=training)
    ledger.append(0, [b"000!"])
    ledger.commit(); ledger.close()
    with pytest.raises(ValueError, match="overlaps"):
        runtime.verify_ledger(ledger.path, [0], meta, k=1, training=training)
    with pytest.raises(ValueError):
        runtime.trial_schedule(tmp_path / "trials.json", "A-prof", "W1", 64, targets, [], 128)


@pytest.mark.metal
def test_ledger_model_resume_and_regeneration(tmp_path):
    run_metal('''
import sys
from pathlib import Path
import numpy as np
from dhi_v6 import runtime, data
root=Path(sys.argv[1])
model=runtime.train(root/"train","P-DISC","A-Q",0,data.synthetic_split(),updates=2,task="synthetic",batch_size=8,checkpoint_every=2,diagnostic_pairs=2)
targets=runtime.trial_schedule(root/"trials.json","A-Q","W1",64,data.synthetic_split()["validation"],[root/"train"],64)
kwargs=dict(stage="A-Q",source="P",method="Main",seed_id=0,pipeline="P-DISC",model=model,task="synthetic",batch=64,checkpoint=runtime.verify_training(root/"train")["checkpoint"])
a,_=runtime.evaluate_block(root/"direct",1,targets,**kwargs)
class Stop:
    n=0
    def check(self):
        self.n+=1
        if self.n==10: raise InterruptedError()
try: runtime.evaluate_block(root/"resumed",1,targets,budget=Stop(),**kwargs)
except InterruptedError: pass
b,_=runtime.evaluate_block(root/"resumed",1,targets,**kwargs)
assert np.array_equal(a,b)
assert (root/"direct/block-1.bin").read_bytes()==(root/"resumed/block-1.bin").read_bytes()
''', tmp_path)


def test_sequential_engine_rules():
    from dhi_v6 import statistics as st
    assert [round(z, 4) for z in (*st.Z_LOOK, st.Z_P, st.Z_CONTRAST, st.Z_S)] == [3.4808, 3.0902, 3.0233, 2.8782, 2.8653, 3.2905]
    estimates = np.array([[0., 0.], [.003, .003]])
    errors = np.array([[.0001, .0001], [.002, .002]])
    stop, decisions = st.joint_decision(estimates, errors, 1)
    assert not stop and decisions == ["REJECTED_BOUNDED", ""]
    stop, decisions = st.joint_decision(estimates, errors, 2, budget_stop=True)
    assert stop and decisions == ["REJECTED_BOUNDED", "NOT_ESTABLISHED_BY_BUDGET"]
    assert st.joint_decision(estimates, errors, 3)[1][-1] == "NOT_ESTABLISHED_UNRESOLVED"
    estimates = np.array([[.02, 0.], [0., .02]])
    assert st.joint_decision(estimates, np.full((2, 2), .0001), 3)[1] == ["REJECTED_NO_CONDITION_GAIN", "REJECTED_NO_RANDOM_ADVANTAGE"]
    assert st.joint_decision([[.001, .001]], [[.00001, .00001]], 1) == (True, ["POSITIVE"])
    z = np.zeros((3, 256), dtype=int)
    outcomes = {p: {"Main": z, "Random": z, "Shuffled": z} for p in st.PIPELINES}
    assert st.stage_c(outcomes, 1)["action"] == "stop"
    outcomes["P-DISC"] = {"Main": z[:, :128], "Random": z[:, :128], "Shuffled": z[:, :128]}
    with pytest.raises(ValueError):
        st.stage_c(outcomes, 1)


def test_design_script_parity():
    import importlib.util
    from dhi_v6 import statistics as st
    path = Path(__file__).parents[1] / "scripts/validate_research_plan_v6.py"
    spec = importlib.util.spec_from_file_location("design_v6", path)
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    generator = data.rng("design-parity")
    est, se = generator.uniform(-.02, .02, (10000, 2)), generator.uniform(0, .005, (10000, 2))
    for z in st.Z_LOOK:
        for final in (False, True):
            expected, upper = design.classify(est, se, z, final)
            actual, actual_upper = st.classify(est, se, z, final)
            assert np.array_equal(actual, expected) and np.array_equal(actual_upper, upper)
    values = ["SUPPORTED", "REJECTED_BOUNDED", "REJECTED_NO_CONDITION_GAIN", "REJECTED_NO_RANDOM_ADVANTAGE",
              "UNTESTABLE", "NOT_ESTABLISHED_UNRESOLVED", "NOT_ESTABLISHED_BY_BUDGET", "NOT_ESTABLISHED_INTEGRITY"]
    for row in generator.choice(values, (10000, 5)):
        assert st.headline(row) == design.headline(row)


def test_replication_p_contrasts_c4_headline():
    from statistics import NormalDist
    from dhi_v6 import statistics as st
    one, zero = np.ones((3, 256), dtype=int), np.zeros((3, 256), dtype=int)
    for entrants in range(1, 6):
        r = st.replication(one, zero, zero, entrants)
        assert r["decision"] == "SUPPORTED" and r["z"] == NormalDist().inv_cdf(1 - .025 / entrants)
        assert st.replication(one, zero, one, entrants)["decision"] == "NOT_ESTABLISHED_NOT_REPLICATED"
    clp = st.probe(np.tile([.9, 1.1], 128))
    assert st.positive_control(one[0], zero[0], zero[0], clp)["GEN_4"]
    assert st.positive_control(one[0], zero[0], one[0], clp)["INFO_4"]
    assert not st.positive_control(one[0], zero[0], one[0], clp)["GEN_4"]
    outcomes = {p: {"Main": zero, "Random": zero, "Shuffled": zero} for p in st.PIPELINES}
    assert len(st.contrasts(outcomes)) == 12
    assert {r["decision"] for r in st.contrasts(outcomes)} == {"EQUIVALENT_WITHIN_DELTA"}
    assert st.compute_advantage("SUPPORTED", 100, 100, 1, 100, 1000, 100)["decision"] == "PER_QUERY_ADVANTAGE"
    result = st.compute_advantage("REJECTED_BOUNDED", 100, 100, 1, 100, 1000000, 100)
    assert result["decision"] == "NO_ADVANTAGE" and result["arithmetic"] == "ARITHMETICALLY_IMPOSSIBLE"
    assert st.cp_bound(8, 30, .025) == pytest.approx(.1227948098723548)
    assert st.cp_bound(8, 30, .025, upper=True) == pytest.approx(.458893651394751)
    assert st.headline(["UNTESTABLE"] * 5) == "FINAL_NOT_ESTABLISHED"
    assert st.headline(["REJECTED_BOUNDED"] * 4 + ["UNTESTABLE"]) == "FINAL_REJECTED_WITH_EXCEPTIONS"


def test_calibration_quick():
    from dhi_v6.statistics import calibrate, cp_bound
    result = calibrate(200)
    assert not result["passed"] and not result["production"]
    assert result["scope"] == "regression-only"
    assert len(result["scenarios"]) == 4
    for values in result["scenarios"].values():
        assert sum(values["stopping_look"].values()) == 200
        assert values["positive_upper"] == cp_bound(values["any_positive"], 200, upper=True)
        assert values["all_rejected_lower"] == cp_bound(values["all_rejected"], 200)
        assert values["planted_lower"] == cp_bound(values["planted_positive"], 200)
    assert result["criteria"] == {
        "null_positive": result["scenarios"]["all_null"]["positive_upper"] <= .025,
        "null_rejected": result["scenarios"]["all_null"]["all_rejected_lower"] >= .95,
        "delta_positive": result["scenarios"]["P-DISC_+delta"]["planted_lower"] >= .95,
    }
    assert max(map(int, result["scenarios"]["budget_after_look2"]["stopping_look"])) <= 2


def test_calibration_production():
    from dhi_v6.statistics import calibrate
    result = calibrate(2000)
    assert result["production"] and result["passed"] and all(result["criteria"].values()), result
    assert result["scope"] == "production" and result["repetitions"] == 2000


# ---------------------------------------------------------------- Phase 7: 예산, 감사, 단계 순서, 보고서, CLI

def test_budget_plan_decisions():
    from dhi_v6.protocol import budget_plan
    q = list(registration()["pipeline_order"])
    profile1 = {"models": {p: {"update_seconds": 0.0, "forward_rows_per_second": 1e12} for p in q},
                "verify_rows_per_second": 1e12}
    qualification = {"Q": q, "settings": {p: {"updates": 40000} for p in q}}

    def plan(cps, a_seconds=0.0, caps=None):
        profile2 = {"models": {p: {"sustained_cps": cps} for p in q}, "random_cps": 1e12}
        return budget_plan(profile1, profile2, qualification, a_seconds, caps=caps)
    fast = plan(1000)
    assert (fast["decision"], fast["block"], fast["p_trials"], fast["stage_s"]) == ("PROCEED", 8192, 4096, True)
    assert 1.5 * fast["estimates"]["8192"]["look2"] <= 80 * 3600
    assert fast["estimates"]["8192"]["regen_block"] == pytest.approx(.01 * fast["estimates"]["8192"]["gen_block"])
    assert plan(220)["block"] == 6144
    halt = plan(100)
    assert halt["decision"] == "HALT" and halt["required_c_hours"] > 80
    approved = {**registration()["caps_hours"], "C": np.ceil(halt["required_c_hours"])}
    assert plan(100, caps=approved)["decision"] == "PROCEED"
    reduced = plan(1000, a_seconds=400000)
    assert (reduced["p_trials"], reduced["stage_s"]) == (2048, False)


def test_audit_v6_rules(tmp_path):
    from dhi_v6.protocol import audit_exposure
    for name, text in {"code/a.py": "print(1)\n", "archive/v5-C.json": "{}\n", "archive/fixture.json": "[]\n",
                       "archive/prefix.csv": "x\n"}.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(text)

    def audit(rows, complete=True):
        path = tmp_path / "inventory.json"
        atomic_json(path, {"schema": "v6-exposure-1", "scope_complete": complete, "scope_boundary": "fixture",
                           "scopes": {"code": ["code"], "archive": ["archive"]}, "reviewed_files": rows,
                           "exposed_groups": {}})
        return audit_exposure(path)

    def row(name, use, roles=(), groups=None):
        return {"path": name, "sha256": file_hash(tmp_path / name), "condition_use": use, "window_roles": list(roles),
                "exposed_groups": groups or {}}
    rows = [row("code/a.py", "none"),
            row("archive/v5-C.json", "window", [{"window": "W2", "rung": 64, "role": r} for r in ("evaluation", "training")]),
            row("archive/fixture.json", "window", [{"window": "W3", "rung": 64, "role": "fixture"},
                                                   {"window": "W4", "rung": 64, "role": "hash-test"}]),
            row("archive/prefix.csv", "prefix", groups={"W3": [5, 6]})]
    result = audit(rows)
    assert result["certified"] and len(result["excluded"]["W2"]) == 4096
    assert result["excluded"]["W3"] == [5, 6] and result["excluded"]["W4"] == []
    rows[2]["window_roles"] = [{"window": "W3", "rung": 64, "role": "evaluation"}]
    exposed = audit(rows)
    assert not exposed["certified"] and len(exposed["excluded"]["W3"]) == 4096
    rows[2]["window_roles"] = []
    assert audit([*rows[:3], {**rows[3], "condition_use": "unreviewed"}])["reason"] == "UNREVIEWED_CLASSIFICATION"
    (tmp_path / "archive/new.npy").write_bytes(b"x")
    assert audit(rows)["reason"] == "UNREVIEWED_FILES"
    assert audit(rows, complete=False)["reason"] == "INCOMPLETE_SCOPE"


def test_select_steps_batch_and_qualification():
    from dhi_v6 import study
    cps = lambda *values: {str(b): {"candidates_per_second": v} for b, v in zip((256, 1024, 2048), values)}
    assert study.choose_batch(cps(100, 100.5, 99)) == 256
    assert study.choose_batch(cps(100, 102, 99)) == 1024
    row = lambda joint, rate: {"normal_joint": joint, "flipped_joint": joint, "duplicate_rate": rate}
    assert study.select_steps({"25": row(240, .02), "50": row(231, .014), "100": row(250, .01)}) == 50
    assert study.select_steps({"25": row(230, .01), "50": row(230, .01), "100": row(230, .01)}) == 100
    assert [study.profile_trials(v) for v in (1, 100000)] == [512, 30208]
    passing = {"updates": 40000, "generation_pass": True, "clp": {"positive": True}}
    rows = {p: [passing] * 3 for p in study.PIPELINES}
    rows["P-DISC"] = [{**passing, "generation_pass": False}] * 3
    rows["R-DISC"] = [{**passing, "generation_pass": False}] * 3
    rows["P-G-CGGE"] = [passing, {**passing, "clp": {"positive": False}}, passing]
    batches = {"25": {"batch": 1024}, "50": {"batch": 256}, "100": {"batch": 256}, "none": {"batch": 2048}}
    profile = {"models": {p: {"by_steps": batches} for p in study.PIPELINES}}
    steps = {p: 25 for p in study.GAUSSIAN}
    first = study.qualification_record([{"updates": 40000, "rows": rows}], steps, profile)
    assert first["repair_state"] == "pending" and first["C1"]["P-DISC"] == "PENDING_REPAIR"
    assert first["Q"] == ["P-G-BGV", "P-G-CGGE", "R-G-BGV"] and first["clp_disabled"]["P-G-CGGE"]
    assert first["settings"]["P-G-BGV"] == {"updates": 40000, "sampling_steps": 25, "batch": 1024}
    repair = {"P-DISC": [{**passing, "updates": 80000}] * 3, "R-DISC": [{**passing, "updates": 80000, "generation_pass": False}] * 3}
    second = study.qualification_record([*first["rounds"], {"updates": 80000, "rows": repair}], steps, profile)
    assert second["repair_state"] == "done" and second["repair_used"]
    assert second["Q"] == ["P-G-BGV", "P-G-CGGE", "P-DISC", "R-G-BGV"] and second["C1"]["R-DISC"] == "UNTESTABLE"
    assert second["settings"]["P-DISC"] == {"updates": 80000, "sampling_steps": None, "batch": 2048}


def _study_root(root):
    from dhi_v6 import study
    root.mkdir(parents=True)
    sealed_json(root / "v6-study.json", {"protocol": PROTOCOL})
    return study


def _terminal_fixture(root, positive=(), untestable=(), replicated=True, stages="CRPS"):
    """Sealed-stage fixture built with the production statistics; no model or MD5 data."""
    from dhi_v6 import statistics as st
    study = _study_root(root)
    q = [p for p in study.PIPELINES if p not in untestable]
    rows = {p: [{"updates": 40000, "normal_joint": 500, "flipped_joint": 498, "generation_pass": p not in untestable,
                 "clp": {"positive": True}}] * 3 for p in study.PIPELINES}
    atomic_json(root / "A.json", {
        "rounds": [{"updates": 40000, "rows": rows}], "Q": q, "repair_used": bool(untestable),
        "C1": {p: "UNTESTABLE" if p in untestable else "PASS" for p in study.PIPELINES},
        "repair_state": "done" if untestable else "not-needed", "sampling_steps": {p: 25 for p in study.GAUSSIAN},
        "clp_disabled": {p: False for p in study.PIPELINES},
        "settings": {p: {"updates": 40000, "sampling_steps": 25 if p in study.GAUSSIAN else None, "batch": 1024} for p in q}})
    zero, one = np.zeros((3, 256), dtype=int), np.ones((3, 256), dtype=int)
    outcomes = {p: {"Main": one if p in positive else zero, "Random": zero, "Shuffled": zero} for p in q}
    final = {**st.stage_c(outcomes, 1, budget_stop=True), "trials": 256, "active": q} if q else None
    metrics = {"rows": 76800, "hits": 10, "valid": 76000, "duplicates": 5, "training_matches": 0, "strict_valid": 70000}
    if "C" in stages and q:
        sealed_json(root / "C.json", {
            "block": 256, "looks": [final], "final": final, "budget_stop": False, "integrity": {}, "clp_partial": False,
            "decisions": {p: final["pipelines"][p]["decision"] for p in q},
            "clp": {p: {"z": .5, "INFO_64": False, "pooled": {"estimate": .01}} for p in q},
            "audit": {p: {"passed": True, "top_1pct_target_hit_share": .05, "hit_only": True} for p in positive},
            "contrasts": st.contrasts(outcomes), "metrics": {p: metrics for p in q}})
    if "R" in stages and positive:
        decision = "SUPPORTED" if replicated else "NOT_ESTABLISHED_NOT_REPLICATED"
        sealed_json(root / "R.json", {"entrants": list(positive), "decisions": {p: decision for p in positive},
                                      "results": {p: st.replication(one, zero, one if not replicated else zero, len(positive))
                                                  for p in positive}, "budget_stop": False, "integrity": {}})
    if "P" in stages:
        clp = st.probe(np.tile([.9, 1.1], 128))
        sealed_json(root / "P.json", {"trials": 4096, "partial": False, "integrity": {}, "pipelines": {
            p: {**st.positive_control(one[0], zero[0], zero[0], clp), "INFO_4_disabled": False,
                "success_at_100": {"Main": 1.0, "Random": 0.0, "MC": 0.0}} for p in q}})
    if "S" in stages:
        sealed_json(root / "S.json", {"skipped": True, "reason": "budget plan omitted Stage S"})
    return study


def test_terminal_report_sentences(tmp_path):
    study = _terminal_fixture(tmp_path / "rejected")
    decision = study.report(tmp_path / "rejected")
    text = (tmp_path / "rejected/FINAL_REPORT_KO.md").read_text()
    assert decision["status"] == "TERMINAL" and decision["headline"] == "FINAL_REJECTED"
    assert "0.5%p 미만으로 배제되었다" in text and "U_R" in text and "## 9. 적용 범위와 일반화" in text
    assert decision["pipelines"]["P-DISC"]["quality"]["duplicates_per_trial"] == 5 / 768
    assert decision["pipelines"]["P-DISC"]["quality"]["strict_valid_rate"] is None
    assert decision["blinding"] == {"procedure": "automatic looks; CLI shows continue/stop only"}
    assert read_json(tmp_path / "rejected/decision.json") == decision
    study = _terminal_fixture(tmp_path / "exceptions", untestable=["R-DISC"])
    assert study.report(tmp_path / "exceptions")["headline"] == "FINAL_REJECTED_WITH_EXCEPTIONS"
    assert "R-DISC(UNTESTABLE; 측정 없음)" in (tmp_path / "exceptions/FINAL_REPORT_KO.md").read_text()
    study = _terminal_fixture(tmp_path / "supported", positive=["P-DISC"])
    decision = study.report(tmp_path / "supported")
    assert decision["headline"] == "FINAL_SUPPORTED" and decision["pipelines"]["P-DISC"]["C3"] == "SUPPORTED"
    assert "W4에서 재현되었다" in (tmp_path / "supported/FINAL_REPORT_KO.md").read_text()
    study = _terminal_fixture(tmp_path / "unreplicated", positive=["P-DISC"], replicated=False)
    assert study.report(tmp_path / "unreplicated")["pipelines"]["P-DISC"]["C3"] == "NOT_ESTABLISHED_NOT_REPLICATED"


def test_partial_report_is_not_rejection(tmp_path):
    for name, kwargs in {"no-p": {"stages": "CRS"}, "no-r": {"positive": ["P-DISC"], "stages": "CPS"},
                         "no-c": {"stages": "PS"}}.items():
        study = _terminal_fixture(tmp_path / name, **kwargs)
        decision = study.report(tmp_path / name)
        text = (tmp_path / name / "FINAL_REPORT_KO.md").read_text()
        assert (decision["status"], decision["headline"]) == ("INCOMPLETE", "NOT_FINAL")
        assert "REJECTED" not in text and "REJECTED" not in json.dumps(decision) and "NOT_FINAL" in text
    root = tmp_path / "budget"
    study = _terminal_fixture(root, stages="")
    sealed_json(root / "failure.json", {"stage": "A", "reason": "NOT_ESTABLISHED_BY_BUDGET"})
    decision = study.report(root)
    assert decision["headline"] == "FINAL_NOT_ESTABLISHED"
    assert {v["C3"] for v in decision["pipelines"].values()} == {"NOT_ESTABLISHED_BY_BUDGET"}
    assert decision["failures"][0]["stage"] == "A"
    root = tmp_path / "empty-q"
    study = _terminal_fixture(root, untestable=list(registration()["pipeline_order"]), stages="")
    assert study.report(root)["headline"] == "FINAL_NOT_ESTABLISHED"


def test_stage_order_and_blinding(tmp_path, monkeypatch, capsys):
    from dhi_v6 import study
    calls = []

    def scenario(decision, audited, q=("P-DISC",)):
        calls.clear()
        root = tmp_path / f"{decision}-{audited}-{len(q)}"
        root.mkdir()
        monkeypatch.setattr(study, "stage_a", lambda r, phase="all": calls.append("A") or atomic_json(
            r / "A.json", {"Q": list(q), "repair_state": "not-needed"}))
        c = {"decisions": {"P-DISC": decision}, "audit": {"P-DISC": {"passed": audited}} if decision == "POSITIVE" else {}}
        monkeypatch.setattr(study, "stage_c", lambda r: calls.append("C") or c)
        for name in "RPS":
            monkeypatch.setattr(study, f"stage_{name.lower()}", lambda r, name=name: calls.append(name))
        monkeypatch.setattr(study, "report", lambda r: calls.append("report") or {"status": "TERMINAL", "headline": "x"})
        study.run_all(root)
        return list(calls)
    assert scenario("POSITIVE", True) == ["A", "C", "R", "P", "S", "report"]
    assert scenario("POSITIVE", False) == ["A", "C", "P", "S", "report"]
    assert scenario("REJECTED_BOUNDED", False) == ["A", "C", "P", "S", "report"]
    assert scenario("REJECTED_BOUNDED", False, q=()) == ["A", "report"]
    monkeypatch.undo()
    root = tmp_path / "order"
    _study_root(root)
    for stage in (study.stage_r, study.stage_p, study.stage_s):
        with pytest.raises(ValueError, match="C.json"):
            stage(root)
    with pytest.raises(ValueError, match="protocol.frozen"):
        study.stage_c(root)
    sealed_json(root / "C.json", {"decisions": {"P-DISC": "POSITIVE"}, "audit": {"P-DISC": {"passed": True}}})
    for stage in (study.stage_p, study.stage_s):
        with pytest.raises(ValueError, match="Stage R"):
            stage(root)
    sealed_json(root / "R.json", {"entrants": ["P-DISC"]})
    with pytest.raises(ValueError, match="P.json"):
        study.stage_s(root)
    root = tmp_path / "blind"
    _study_root(root)
    folder = root / "C/eval/P-DISC/Main-0"
    atomic_json(folder / "block-1.json", {"rows": 819200, "elapsed_seconds": 100.0, "hits": 3000, "success_at_100": 20,
                                         "success_at_1": 1, "top_1pct_target_hit_share": .5})
    atomic_json(folder / "block-1.commit.json", {"committed_rows": 819200})
    atomic_json(root / "C/looks/look-1.json", {"action": "continue", "pipelines": {"P-DISC": {"decision": "POSITIVE",
                                                                                              "estimate": .01}}})
    capsys.readouterr()
    assert study.main(["status", "--root", str(root)]) == 0
    printed = capsys.readouterr().out
    shown = json.loads(printed)
    assert shown["streams"]["C"] == {"completed_stream_blocks": 1, "committed_rows": 819200, "recent_rows_per_second": 8192.0}
    for word in ("hits", "success", "estimate", "lower", "upper", "POSITIVE", "REJECTED", "continue", "share"):
        assert word not in printed


def test_cli_preconditions(tmp_path, capsys):
    from dhi_v6 import study
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("not a study")
    with pytest.raises(SystemExit):
        study.main(["run", "--root", str(other), "--stage", "A"])
    root = tmp_path / "study"
    with pytest.raises(SystemExit):
        study.main(["run", "--root", str(root), "--stage", "C", "--phase", "impl"])
    with pytest.raises(ValueError, match="A-impl"):
        study.main(["run", "--root", str(root), "--stage", "A", "--phase", "prof1"])
    assert read_json(root / "v6-study.json") == {"protocol": PROTOCOL}
    with pytest.raises(ValueError, match="HALT"):
        study.main(["approve-caps", "--root", str(root), "--stage", "C", "--hours", "120", "--reason", "fixture"])
    atomic_json(root / "halt.json", {"decision": "HALT"})
    with pytest.raises(ValueError, match="increase"):
        study.main(["approve-caps", "--root", str(root), "--stage", "C", "--hours", "80", "--reason", "not an increase"])
    assert study.main(["approve-caps", "--root", str(root), "--stage", "C", "--hours", "120", "--reason", "fixture"]) == 0
    from dhi_v6.protocol import effective_caps
    assert [effective_caps(root)[k] for k in ("C", "required", "required_with_repair")] == [120, 154, 166]
    (root / "C").mkdir()
    with pytest.raises(ValueError, match="MD5"):
        study.main(["approve-caps", "--root", str(root), "--stage", "C", "--hours", "130", "--reason", "again"])
    assert study.main(["report", "--root", str(root)]) == 0
    assert read_json(root / "decision.json")["headline"] == "NOT_FINAL"
    atomic_json(root / "failure.json", {"stage": "A", "reason": "NOT_ESTABLISHED_BY_BUDGET"})
    with pytest.raises(RuntimeError, match="terminal failure"):
        study.main(["run", "--root", str(root), "--stage", "all"])
    capsys.readouterr()
    assert study.main(["plan"]) == 0
    assert json.loads(capsys.readouterr().out) == registration()


@pytest.mark.metal
def test_stage_a_paths_small(tmp_path):
    run_metal('''
import sys
from pathlib import Path
import numpy as np
from dhi_v6 import data, runtime, study
from dhi_v6.protocol import read_json
root = Path(sys.argv[1])
class Free:
    def check(self):
        pass
folder = study.a_q_folder(root, "P-DISC", 0, 4)
runtime.train(folder, "P-DISC", "A-Q", 0, data.synthetic_split(), updates=4, task="synthetic", batch_size=8,
              checkpoint_every=2, diagnostic_pairs=2)
row = study.qualify_seed(root, "P-DISC", 0, 4, None, 2048, Free())
assert row["batch"] == 512 and row["md5_calls"] == 0 and not row["generation_pass"]
assert row["normal_valid"] == row["flipped_valid"] == 512 and row["clp"]["threshold"] == 3.26
assert read_json(folder / "qualification.json") == row == study.qualify_seed(root, "P-DISC", 0, 4, None, 2048, Free())
try:
    study.qualify_seed(root, "P-DISC", 0, 4, 25, 2048, Free())
    raise AssertionError("changed steps accepted")
except ValueError:
    pass

def fake(model, labels, keys, steps, batch):
    return [b"%03X" % int(y) + b"ZZZZ" for y in labels], None, None
rows = study.dev_rows(None, "P-G-BGV", Free(), generator=fake)
assert rows["25"]["normal_joint"] == rows["25"]["flipped_joint"] == 256 and rows["25"]["duplicate_rate"] == .99
assert study.select_steps(rows) == 25

study._TRAINING_FIXTURE[:] = [np.sort(np.frombuffer(data.rng("fixture").bytes(16 * 1000), dtype="V16"))]
model, _ = study.load_trained(folder, "P-DISC", "A-Q", 0)
result = study.sustained(root / "prof", Free(), pipeline="P-DISC", model=model, batch=256, cps=1, warmup=0, measure=0)
assert result["blocks"] == 1 and result["candidates"] == 51200 and result["sustained_cps"] > 0
assert not (root / "prof").exists()
random = study.sustained(root / "random", Free(), source="R", warmup=0, measure=0)
assert random["candidates"] == 204800 and random["sustained_cps"] > 0
''', tmp_path)


@pytest.mark.metal
def test_quick_check_cli(tmp_path):
    run_metal('''
import sys
from pathlib import Path
from dhi_v6 import study
from dhi_v6.protocol import read_json
root = Path(sys.argv[1])
assert study.main(["check", "--root", str(root), "--quick"]) == 0
result = read_json(root / "quick-check.json")
assert result["quick"] and not result["passed"] and result["quick_paths_passed"] and not result["certifies_study"]
assert not (root / "A-impl.json").exists()
''', tmp_path / "quick")


@pytest.mark.metal
def test_md5_stage_flow_on_validation_fixture(tmp_path):
    """C/R/P/S runners end to end at fixture scale. Every split's test pool is replaced by its validation groups."""
    run_metal('''
import sys
from pathlib import Path
import numpy as np
from dhi_v6 import data, protocol, study
from dhi_v6 import statistics as st
from dhi_v6.protocol import BudgetExceeded, atomic_json, freeze, read_json, sealed_json, source_manifest

real_split, real_stage_c = data.split, st.stage_c
def split(window, rung, excluded=()):
    groups = real_split(window, rung, excluded)
    return {"test": groups["validation"], "validation": groups["validation"], "train": groups["train"]}
data.split = split
study.RANDOM_BATCH = 64
study.REG["stage_c"]["clp_pairs_per_seed"] = 64
study.REG["stage_r"]["trials"] = 64
study.REG["stage_p"]["clp_pairs"] = 64
study.REG["stage_s"].update(model="D1-S", updates=2, trials=64, clp_pairs=64)

def fixture(root):
    root.mkdir(parents=True)
    sealed_json(root / "v6-study.json", {"protocol": study.PROTOCOL})
    settings = {"P-DISC": {"updates": 2, "sampling_steps": None, "batch": 64}}
    by_steps = {"none": {"batch": 64, "candidates_per_second": 50000.0}}
    atomic_json(root / "A-impl.json", {"passed": True, "quick": False, "source": source_manifest(),
                                       "gates": {"G9": {"production": True, "passed": True, "repetitions": 2000}}})
    atomic_json(root / "A-prof-1.json", {"models": {p: {"by_steps": by_steps, "update_seconds": .01} for p in study.PIPELINES},
                                         "scale": {"by_steps": by_steps}, "md5_per_second": 1e6,
                                         "verify_rows_per_second": 1e6})
    atomic_json(root / "A-prof-2.json", {"fixture": True})
    atomic_json(root / "A-dev.json", {"fixture": True})
    atomic_json(root / "A.json", {"Q": ["P-DISC"], "C1": {p: "PASS" if p == "P-DISC" else "UNTESTABLE" for p in study.PIPELINES},
                                  "repair_state": "done", "repair_used": False, "settings": settings, "rounds": [],
                                  "clp_disabled": {p: False for p in study.PIPELINES}})
    atomic_json(root / "budget-plan.json", {"decision": "PROCEED", "block": 64, "p_trials": 64, "stage_s": True,
                                            "pipelines": settings})
    atomic_json(root / "exposure-audit.json", {"certified": True, "excluded": {"W1": [], "W2": list(range(4096)),
                                                                               "W3": [], "W4": []}})
    freeze(root)
    return root

# Budget exhausted during block 2: look 1 becomes final, CLP_64 is marked partial.
class Limited(protocol.Budget):
    def check(self):
        super().check()
        if self.stage == "C" and (self.root / "C/looks/look-1.json").exists():
            raise BudgetExceeded("fixture cap")
study.Budget = Limited
root = fixture(Path(sys.argv[1]) / "budget")
c = study.stage_c(root)
assert c["budget_stop"] and c["final"]["look"] == 1 and c["final"]["budget_stop"] and c["clp_partial"]
assert c["decisions"]["P-DISC"] in ("NOT_ESTABLISHED_BY_BUDGET", "REJECTED_BOUNDED", "REJECTED_NO_CONDITION_GAIN",
                                    "REJECTED_NO_RANDOM_ADVANTAGE", "POSITIVE")
assert [look["action"] for look in c["looks"]] == ["continue"] and not (root / "C/looks/look-2.json").exists()
study.Budget = protocol.Budget

# Forced POSITIVE at look 1: artifact audit, CLP_64, R entry, P, S and the terminal report.
def forced(outcomes, look, *, budget_stop=False):
    result = real_stage_c(outcomes, look, budget_stop=budget_stop)
    result["pipelines"]["P-DISC"]["decision"] = "POSITIVE"
    return {**result, "action": "stop"}
st.stage_c = forced
root = fixture(Path(sys.argv[1]) / "positive")

def exhaust(stage):
    state = read_json(root / "budget.json")
    state["seconds"][stage] = protocol.effective_caps(root)[stage] * 3600 + 1
    atomic_json(root / "budget.json", state)

# A crash after the looks and CLP probes were sealed, then a rerun whose C session cannot open.
real_metrics = study.stream_metrics
def crash(*args, **kwargs):
    raise RuntimeError("fixture crash before C.json")
study.stream_metrics = crash
try:
    study.stage_c(root)
    raise AssertionError("fixture crash not raised")
except RuntimeError as error:
    assert "fixture crash" in str(error)
study.stream_metrics = real_metrics
assert (root / "C/looks/look-1.json").exists() and not (root / "C.json").exists()
exhaust("C")
c = study.stage_c(root)
assert c["final"]["look"] == 1 and not c["budget_stop"] and not c["clp_partial"]
audit = c["audit"]["P-DISC"]
assert c["decisions"] == {"P-DISC": "POSITIVE"} and audit["passed"]
assert audit["successful_payloads"] == c["metrics"]["P-DISC"]["hits"] and audit["rehash_mismatches"] == 0
assert c["clp"]["P-DISC"]["pooled"]["trials"] == 3 * 64 and audit["INFO_64"] == c["clp"]["P-DISC"]["INFO_64"]
assert study.stage_c(root) == c and study.replication_entrants(c) == ["P-DISC"]
r = study.stage_r(root)
assert r["entrants"] == ["P-DISC"] and r["decisions"]["P-DISC"] in ("SUPPORTED", "NOT_ESTABLISHED_NOT_REPLICATED")
p = study.stage_p(root)
assert not p["partial"] and set(p["pipelines"]["P-DISC"]) >= {"GEN_4", "INFO_4", "comparisons", "success_at_100"}
s = study.stage_s(root)
assert not s["skipped"] and not s["partial"] and set(s["success_at_100"]) == {"Main", "MC", "Random"}
assert (root / "S/runs/P-DISC-D1-S/Main-0/u2/complete.json").exists()
decision = study.report(root)
assert decision["status"] == "TERMINAL" and decision["pipelines"]["P-DISC"]["C3"] == r["decisions"]["P-DISC"]
assert decision["pipelines"]["P-DISC"]["C4"]["rho"] == 20.0 and decision["pipelines"]["P-DISC"]["audit"]["passed"]
text = (root / "FINAL_REPORT_KO.md").read_text()
assert "Stage R(W4 재현)" in text and "## 8. Stage S" in text
status = study.status(root)
assert status["c_looks_sealed"] == 1 and status["streams"]["C"]["completed_stream_blocks"] == 9
assert status["streams"]["R"]["committed_rows"] == 9 * 6400 and not status["failure"] and status["budget_stops"] == []

# With every cap exhausted at session start, R/P/S reassemble the same results from sealed data.
for stage in ("R", "P", "S"):
    previous = read_json(root / f"{stage}.json")
    (root / f"{stage}.json").unlink()
    (root / f"{stage}.json.sha256").unlink()
    exhaust(stage)
    assert getattr(study, f"stage_{stage.lower()}")(root) == previous, stage
''', tmp_path)


def test_stage_a_orchestration(tmp_path, monkeypatch, capsys):
    """Phase order, repair budget, HALT and cap approval, and Stage A budget exhaustion with fake phase work."""
    from dhi_v6 import study
    from dhi_v6.protocol import BudgetExceeded, effective_caps
    calls = []
    q = list(study.PIPELINES)
    speed = {"cps": 1000.0}
    by_steps = {"25": {"batch": 1024, "candidates_per_second": 1e5}, "none": {"batch": 2048, "candidates_per_second": 1e5}}

    def once(name, value):
        return lambda r: (r / name).exists() or atomic_json(r / name, value(r) if callable(value) else value)
    writes = {
        "impl": once("A-impl.json", lambda r: {"passed": True, "quick": False, "source": source_manifest(),
                                               "gates": {"G9": {"production": True, "passed": True, "repetitions": 2000}}}),
        "prof1": once("A-prof-1.json", {"models": {p: {"update_seconds": 0.0, "forward_rows_per_second": 1e12,
                                                       "by_steps": by_steps} for p in q},
                                        "verify_rows_per_second": 1e12, "md5_per_second": 1e6}),
        "train": lambda r: None,
        "dev": once("A-dev.json", {"pipelines": {}}),
        "evaluate": once("A.json", lambda r: {"Q": q[:-1], "repair_state": "pending", "repair_used": False,
                                              "C1": {**{p: "PASS" for p in q[:-1]}, q[-1]: "PENDING_REPAIR"},
                                              "settings": {p: {"updates": 40000} for p in q[:-1]}}),
        "repair": lambda r: atomic_json(r / "A.json", {**read_json(r / "A.json"), "Q": q, "repair_state": "done",
                                                       "repair_used": True, "C1": {p: "PASS" for p in q},
                                                       "settings": {p: {"updates": 40000} for p in q}}),
        "prof2": lambda r: atomic_json(r / "A-prof-2.json", {"models": {p: {"sustained_cps": speed["cps"]} for p in q},
                                                             "random_cps": 1e12}),
    }

    def fake(name):
        def run(r, budget):
            calls.append((name, budget.stage))
            if name == "train" and speed.get("exhaust"):
                raise BudgetExceeded("fixture cap")
            writes[name](r)
        return run
    for name in writes:
        monkeypatch.setattr(study, f"phase_{name}", fake(name))
    root = tmp_path / "study"
    audit = {"certified": True, "excluded": {"W1": [], "W2": list(range(4096)), "W3": [], "W4": []}}

    speed["cps"] = 100.0
    root.mkdir()
    sealed_json(root / "v6-study.json", {"protocol": PROTOCOL})
    atomic_json(root / "exposure-audit.json", audit)
    assert study.main(["run", "--root", str(root), "--stage", "A"]) == 3
    assert [c for c in calls] == [("impl", "A"), ("prof1", "A"), ("train", "A"), ("dev", "A"), ("evaluate", "A"),
                                  ("repair", "A_repair"), ("prof2", "A")]
    assert read_json(root / "budget-plan.json")["decision"] == "HALT" and (root / "halt.json").exists()
    assert study.status(root)["halt"] and "A_repair" in read_json(root / "budget.json")["seconds"]
    hours = float(np.ceil(read_json(root / "budget-plan.json")["required_c_hours"]))
    assert study.main(["approve-caps", "--root", str(root), "--stage", "C", "--hours", str(hours), "--reason", "fixture"]) == 0
    calls.clear()
    assert study.main(["run", "--root", str(root), "--stage", "A"]) == 0
    assert ("repair", "A_repair") not in calls
    frozen = read_json(root / "protocol.frozen.json")
    assert frozen["caps"]["C"] == hours == effective_caps(root)["C"] and "cap-override.json" in frozen["artifacts"]
    assert frozen["settings"]["decision"] == "PROCEED" and frozen["Q"] == q and not study.status(root)["halt"]
    assert study.stage_a(root) == frozen

    speed.update(cps=1000.0, exhaust=True)
    calls.clear()
    root = tmp_path / "exhausted"
    root.mkdir()
    sealed_json(root / "v6-study.json", {"protocol": PROTOCOL})
    assert study.main(["run", "--root", str(root), "--stage", "A"]) == 2
    failure = read_json(root / "failure.json")
    assert (failure["stage"], failure["phase"], failure["reason"]) == ("A", "train", "NOT_ESTABLISHED_BY_BUDGET")
    decision = read_json(root / "decision.json")
    assert decision["status"] == "TERMINAL" and decision["headline"] == "FINAL_NOT_ESTABLISHED"
    assert "A_repair" not in read_json(root / "budget.json")["seconds"]
    with pytest.raises(RuntimeError, match="terminal failure"):
        study.main(["run", "--root", str(root), "--stage", "A"])


@pytest.mark.metal
def test_profiling_helpers_small(tmp_path):
    run_metal('''
import sys
from pathlib import Path
from dhi_v6 import study
root = Path(sys.argv[1])
class Free:
    def check(self):
        pass
study.REG["profiling"].update(train_warmup_updates=1, train_timed_updates=2, burst_seconds=0.05)
study.REG["generation"]["batch_options"] = [64, 128]
study.FORWARD_SECONDS = 0.05
for pipeline, architecture in (("P-G-BGV", None), ("R-G-BGV", None), ("P-G-CGGE", None), ("R-DISC", None), ("P-DISC", "D1-T-L")):
    model, seconds = study._timed_updates(pipeline, architecture, root / "scratch", Free())
    assert seconds > 0 and study._forward_rows(model, pipeline) > 0
    options = [25] if pipeline in study.GAUSSIAN else [None]
    profile = study._generation_profile(model, pipeline, options, Free())
    row = profile["25" if pipeline in study.GAUSSIAN else "none"]
    assert set(row["batches"]) == {"64", "128"} and row["batch"] in (64, 128) and row["candidates_per_second"] > 0
    assert all(v["peak_memory"] > 0 for v in row["batches"].values())
assert study._md5_rate() > 0
written, verified = study._ledger_rates(root / "ledger", Free(), trials=512)
assert written > 0 and verified > 0
''', tmp_path)


def test_cap_override_extends_required_path(tmp_path):
    from dhi_v6.protocol import Budget, BudgetExceeded, approve_caps, effective_caps
    atomic_json(tmp_path / "halt.json", {"decision": "HALT"})
    approve_caps(tmp_path, "C", 120, "fixture")
    caps = effective_caps(tmp_path)
    assert (caps["A"], caps["C"], caps["P"], caps["required"], caps["required_with_repair"]) == (24, 120, 10, 154, 166)
    # 20 h of A plus 95 h of C stays inside the extended required path; the C cap itself still binds.
    atomic_json(tmp_path / "budget.json", {"seconds": {"A": 20 * 3600, "C": 95 * 3600}, "sessions": []})
    Budget(tmp_path, "C").flush()
    atomic_json(tmp_path / "budget.json", {"seconds": {"A": 20 * 3600, "C": 120 * 3600 + 1}, "sessions": []})
    with pytest.raises(BudgetExceeded):
        Budget(tmp_path, "C")
    assert effective_caps(tmp_path / "no-override")["required"] == 114
