"""Runnable implementation gates. Quick checks never certify production gates."""
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from .data import (LADDER, decode, digest_batch, digest_reference, encode, fresh_batch,
                   hash_batch, hash_one, key_words, messages, prior_candidates, rng,
                   source, split, synthetic_split, valid, window_value)
from .protocol import sealed_json
from .runtime import Ledger, evaluate, load_checkpoint, train
from .statistics import P0, calibrate, stage_c


def hash_gate(count=100000):
    payload, lengths = source(rng("hash-gate"), count)
    raw = messages(payload, lengths)
    for rung in (*LADDER, 64):
        vector = digest_batch(payload, lengths, rung)
        for row, message in zip(vector, raw, strict=True):
            reference = digest_reference(message, rung)
            if row.tobytes() != reference:
                raise AssertionError(f"Scalar/vector MD5 disagreement at r={rung}")
            if rung == 64 and reference != hashlib.md5(message).digest():
                raise AssertionError("Full MD5 differs from hashlib")
        for window in ("W1", "W2", "W3"):
            expected = np.array([window_value(row.tobytes(), window) for row in vector])
            assert np.array_equal(hash_batch(payload, lengths, rung, window), expected)
    changed, changed_lengths = source(rng("r4-suffix"), count)
    changed[:, :4] = payload[:, :4]
    assert np.array_equal(hash_batch(changed, changed_lengths, 4), hash_batch(payload, lengths, 4))
    for message in (b"", b"a", b"abc", b"message digest", b"abcdefghijklmnopqrstuvwxyz", b"x"*55):
        assert digest_reference(message) == hashlib.md5(message).digest()
    assert window_value(bytes(16), "W1") == 0
    assert decode(encode(payload, lengths)) == raw
    return {"messages": count, "rungs": list(LADDER)+[64], "windows": ["W1", "W2", "W3"], "passed": True}


def sampler_gate(count=4096):
    from .models import candidate_keys, make_model, parameter_count, sample, sample_reference
    labels = rng("sampler-gate").integers(0, 4096, count)
    keys = candidate_keys(("sampler-gate",), np.arange(count))
    model = make_model("D1-S", ("sampler-gate",))
    assert parameter_count(model) == 508668
    reference = sample_reference(model, labels, keys)
    for batch in (64, 1024):
        actual = np.concatenate([sample(model, labels[o:o+batch], keys[o:o+batch]) for o in range(0, count, batch)])
        assert np.array_equal(reference, actual), f"Sampler changed with batch={batch}"
    assert all(valid(x) for x in decode(reference))
    # The same gate also exercises both new Transformer sizes.
    for architecture in ("D1-T", "D1-T-L"):
        model = make_model(architecture, ("sampler-gate", architecture))
        n = min(count, 8)
        assert np.array_equal(sample(model, labels[:n], keys[:n]), sample_reference(model, labels[:n], keys[:n]))
    return {"candidates": count, "batches": [1, 64, 1024], "passed": True, "reference": "independent-v5-scalar"}


def resume_gate(folder, architecture="D1-S", batch_size=8):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .models import candidate_keys, clp_pairs, make_model
    groups = synthetic_split()
    namespace = ("resume-fixture",)
    folder.mkdir(parents=True, exist_ok=True)
    options = dict(updates=4, task="synthetic", batch_size=batch_size, checkpoint_every=2, diagnostic_pairs=4)
    continuous = train(folder/"continuous", architecture, namespace, groups, **options)

    class Interrupt:
        def __init__(self): self.calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 4:
                raise InterruptedError("Injected interruption after saved optimizer state")

    try:
        train(folder/"resumed", architecture, namespace, groups, budget=Interrupt(), **options)
        raise AssertionError("Interruption did not fire")
    except InterruptedError:
        pass
    resumed = train(folder/"resumed", architecture, namespace, groups, **options)
    for (ka, a), (kb, b) in zip(tree_flatten(continuous.parameters()), tree_flatten(resumed.parameters()), strict=True):
        assert ka == kb and np.array_equal(np.asarray(a), np.asarray(b)), ka
    data = fresh_batch(namespace, 19, 16, groups["train"], task="synthetic")
    again = fresh_batch(namespace, 19, 16, groups["train"], task="synthetic")
    assert all(np.array_equal(a, b) for a, b in zip(data, again, strict=True))
    for window in ("W1", "W2", "W3"):
        ownership = split("gate", window, 64)
        p, n, labels, _ = fresh_batch(namespace, 0, 256, ownership["train"], window=window)
        assert np.isin(hash_batch(p, n, window=window), ownership["train"]).all()
    payload, lengths, labels, _ = data
    keys = candidate_keys(namespace, np.arange(16))
    d = clp_pairs(resumed, encode(payload, lengths), lengths, labels, keys)
    flipped = labels.reshape(-1, 2)[:, ::-1].reshape(-1)
    opposite = clp_pairs(resumed, encode(payload, lengths), lengths, flipped, keys)
    assert np.allclose(d, -opposite, rtol=0, atol=0)
    # Explicit checkpoint restore works without global RNG state.
    restored = make_model(architecture, ("different-init",))
    assert load_checkpoint(folder/"continuous", restored) == 4
    mx.eval(restored.parameters())
    targets = np.array(groups["acceptance"][:4])
    kwargs = dict(method="Main", seed_id=0, task="synthetic", k=3)
    evaluate(folder/"eval-full", continuous, namespace, targets, batch_size=4, **kwargs)
    class StopSampling:
        def __init__(self): self.calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 2:
                raise InterruptedError("Injected partial trial interruption")
    try:
        evaluate(folder/"eval-resume", continuous, namespace, targets, batch_size=4, budget=StopSampling(), **kwargs)
        raise AssertionError("Sampling interruption did not fire")
    except InterruptedError:
        pass
    evaluate(folder/"eval-resume", resumed, namespace, targets, batch_size=3, **kwargs)
    import sqlite3
    rows = []
    for name in ("eval-full", "eval-resume"):
        with sqlite3.connect(folder/name/"candidates.sqlite") as db:
            rows.append(db.execute("SELECT * FROM candidates ORDER BY trial,attempt").fetchall())
    assert rows[0] == rows[1]
    return {"passed": True, "architecture": architecture, "batch_size": batch_size,
            "optimizer_resume_bitwise": True, "candidate_resume_bitwise": True, "clp_antisymmetry": True}


def ledger_gate(folder):
    namespace = ("ledger-gate",)
    meta = {"task": "synthetic", "rung": 64, "window": "W1", "method": "Main", "rng_namespace": list(namespace)}
    targets = np.array([0, 0, 4095])
    payloads = [None, b"000!", b"000!", b"FFFF", b"000!", b"FFFF", b"FFFF", b"FFFF", b"000!"]
    keys = key_words(namespace, np.arange(9))
    path = folder / "ledger.sqlite"
    ledger = Ledger(path, meta, targets, 3)
    ledger.append(0, payloads[:4], keys[:4])
    ledger.close()
    ledger = Ledger(path, meta, targets, 3)
    ledger.append(4, payloads[4:], keys[4:])
    outcome, result = ledger.verify()
    assert outcome.tolist() == [1, 1, 1] and result["rows"] == 9 and result["hits"] == 5
    assert result["valid"] == 8 and result["duplicate_count"] == 6
    try:
        ledger.append(4, payloads[4:], keys[4:])
        raise AssertionError("Duplicate commit accepted")
    except ValueError:
        pass
    with ledger.db:
        ledger.db.execute("UPDATE candidates SET hit=1 WHERE trial=0 AND attempt=0")
    try:
        ledger.verify()
        raise AssertionError("Tampered hit accepted")
    except ValueError:
        pass
    ledger.close()
    # Candidate identities survive changed batch boundaries and a partial trial.
    full = evaluate(folder/"random-full", None, namespace, targets, method="Random", seed_id=0, task="synthetic", k=3, batch_size=4)
    ns = (*namespace, "batch-prior")
    a = prior_candidates(ns, np.arange(20))
    b = [prior_candidates(ns, np.arange(o, min(o+3, 20))) for o in range(0, 20, 3)]
    assert all(np.array_equal(a[i], np.concatenate([x[i] for x in b])) for i in (0, 1))
    return {"passed": True, "invalid_and_duplicates_consume_attempts": True, "full_budget_after_success": True}


def planted_fixture(folder, trials=65536, k=100, budget=None):
    """Full ledger path with validation-only preimages, never a study test pool.

    Coupled prior candidates give the +0 fixture exact equality. Independent
    candidate identities remain recorded. A fixed fraction of Main trials have
    one candidate replaced with a validation preimage for the +delta fixture.
    """
    folder = Path(folder)
    groups = split("planted-validation-only", "W2", 64)["validation"]
    target = int(groups[0])
    table = None
    for update in range(1000):
        payload, lengths = source(rng("planted-table", update), 4096)
        labels = hash_batch(payload, lengths, window="W2")
        hits = np.flatnonzero(labels == target)
        if len(hits):
            i = hits[0]
            table = payload[i, :lengths[i]].tobytes()
            break
    assert table is not None
    outcomes = {method: [] for method in ("Random", "Shuffled", "Main")}
    for seed_id in range(3):
        prior_namespace = ("planted-prior", seed_id)
        targets = np.full(trials, target, dtype=np.int32)
        selected = set(rng("planted-trials", seed_id).choice(trials, int(round(.0025/(1-P0)*trials)), replace=False).tolist())
        for method in outcomes:
            namespace = ("planted", method, seed_id)
            meta = {"task": "md5", "window": "W2", "rung": 64, "method": method, "rng_namespace": list(namespace), "fixture": True}
            ledger = Ledger(folder/f"{method}-{seed_id}.sqlite", meta, targets, k)
            for offset in range(ledger.count(), trials*k, 8192):
                if budget:
                    budget.check()
                indices = np.arange(offset, min(offset+8192, trials*k))
                p, n = prior_candidates(prior_namespace, indices)
                payloads = messages(p, n)
                if method == "Main":
                    for i, index in enumerate(indices):
                        if index % k == 0 and int(index//k) in selected:
                            payloads[i] = table
                ledger.append(offset, payloads, key_words(namespace, indices))
            values, _ = ledger.verify(budget=budget)
            outcomes[method].append(values)
            ledger.close()
    arrays = {k: np.stack(v) for k, v in outcomes.items()}
    null = stage_c(arrays["Random"], arrays["Random"], arrays["Shuffled"])
    lift = stage_c(arrays["Main"], arrays["Random"], arrays["Shuffled"])
    assert null["decision"] == "REJECTED_BOUNDED"
    assert lift["decision"] == "POSITIVE"
    return {"passed": trials == 65536 and k == 100, "trials": trials, "k": k, "window": "W2", "validation_target": target,
            "null": null, "lift": lift, "fixture_only": True}


def implementation_gate(root, quick=False, budget=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v5-checks-", dir=root) as temporary:
        folder = Path(temporary)
        checks = {"hash": hash_gate(64 if quick else 100000), "sampler": sampler_gate(8 if quick else 4096),
                  "ledger": ledger_gate(folder)}
        for architecture in (("D1-S",) if quick else ("D1-S", "D1-T", "D1-T-L")):
            checks[f"resume-{architecture}"] = resume_gate(folder/architecture, architecture, 8 if quick else 256)
        checks["calibration"] = calibrate(100 if quick else 20000)
        if not quick:
            checks["planted_lift"] = planted_fixture(folder / "planted", budget=budget)
    return {"passed": not quick and all(c["passed"] for c in checks.values()), "quick": quick, "checks": checks,
            "study_md5_condition_pairs": 0, "hash_calls_scope": "implementation-fixtures-only", "sampler_gate_amendment": "new scalar oracle replaces legacy model parity"}
