"""A-impl 게이트 G1–G10. 틀은 dhi_v5/checks.py를 따르며, quick 모드는 경로만 확인하고 연구 PASS를 만들지 않는다."""
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from . import PROTOCOL, codecs, data
from . import statistics as st
from .protocol import read_json, registration

FONT_SHA256 = "6ef6d0bfa6e29c823ad9ff7d6b4b29b9af5f739fee711268ff5b83ca7aeed50a"
RFC_VECTORS = (b"", b"a", b"abc", b"message digest", b"abcdefghijklmnopqrstuvwxyz", b"x" * 55)


def sizes(quick):
    reg = registration()
    return {"hash_messages": 1000 if quick else reg["hash_gate"]["messages_per_source"],
            "d1_candidates": 64 if quick else 4096,
            "g3_candidates": 128 if quick else 2048,
            "g3_batches": [64, 128] if quick else [64, 256, 1024, 2048],
            "g3_reference": 4 if quick else 256,
            "g3_steps": [25] if quick else reg["models"]["G3-U"]["sampling_steps_options"],
            "calibration": 200 if quick else reg["calibration_repetitions"],
            "planted_trials": 1024 if quick else reg["stage_c"]["looks"] * reg["stage_c"]["block"]}


def hash_gate(count):
    rungs = registration()["hash_gate"]["rungs"]
    for src in data.SOURCES:
        payload, lengths = data.source(data.rng("A-impl", "hash", src), count, src)
        raw = data.messages(payload, lengths)
        for rung in rungs:
            vector = data.digest_batch(payload, lengths, rung)
            for row, message in zip(vector, raw, strict=True):
                reference = data.digest_reference(message, rung)
                if row.tobytes() != reference:
                    raise AssertionError(f"Scalar/vector MD5 disagreement: source={src}, r={rung}")
                if rung == 64 and reference != hashlib.md5(message).digest():
                    raise AssertionError("Full MD5 differs from hashlib")
            for window in data.WINDOWS:
                expected = np.array([data.window_value(row.tobytes(), window) for row in vector])
                if not np.array_equal(data.hash_batch(payload, lengths, rung, window), expected):
                    raise AssertionError(f"Window extraction mismatch: {window}, r={rung}")
        changed, changed_lengths = data.source(data.rng("A-impl", "r4-suffix", src), count, src)
        changed[:, :4] = payload[:, :4]
        if not np.array_equal(data.hash_batch(changed, changed_lengths, 4), data.hash_batch(payload, lengths, 4)):
            raise AssertionError("r=4 W1 depends on payload bytes beyond 0-3")
    for message in RFC_VECTORS:
        if data.digest_reference(message) != hashlib.md5(message).digest():
            raise AssertionError("RFC 1321 vector mismatch")
    return {"passed": True, "messages_per_source": count, "rungs": rungs, "windows": list(data.WINDOWS)}


def _corpus(src):
    """Every symbol appears at every position for every length 4..31."""
    spec = data.SOURCES[src]
    symbols = np.arange(spec["byte_min"], spec["byte_max"] + 1)
    rows, lengths = [], []
    for length in range(4, 32):
        for offset in range(len(symbols)):
            row = np.zeros(31, dtype=np.uint8)
            row[:length] = symbols[(offset + np.arange(length)) % len(symbols)]
            rows.append(row)
            lengths.append(length)
    return np.stack(rows), np.array(lengths, dtype=np.int32)


def codec_gate():
    if codecs.glyph_table_checksum() != FONT_SHA256:
        raise AssertionError("CGGE glyph table changed")
    counts = {}
    for src in data.SOURCES:
        payload, lengths = _corpus(src)
        messages = data.messages(payload, lengths)
        if data.decode(data.encode(payload, lengths, src), src) != messages:
            raise AssertionError(f"Token round trip failed: {src}")
        counts[f"{src}-tokens"] = len(messages)
        for representation in (("bgv", "cgge") if src == "P" else ("bgv",)):
            images = codecs.encode(payload, lengths, representation, src)
            decoded, margins, strict = codecs.decode(images, lengths, representation, src)
            if decoded != messages or np.any(margins != 0) or not strict.all():
                raise AssertionError(f"Prototype round trip failed: {src}-{representation}")
            if codecs.strict_decode(images, lengths, representation, src) != messages:
                raise AssertionError(f"Strict round trip failed: {src}-{representation}")
            fixed, mask = codecs.structure(np.array([4]), representation)
            tied, _, _ = codecs.decode(np.where(mask, 0., fixed).astype(np.float32), [4], representation, src)
            if tied != [bytes([data.SOURCES[src]["byte_min"]]) * 4]:
                raise AssertionError(f"Prototype tie rule changed: {src}-{representation}")
            counts[f"{src}-{representation}"] = len(messages)
    bits = codecs.BYTE_BITS[33:127]
    hamming = np.abs(bits[:, None] - bits[None]).sum(-1) + np.eye(len(bits)) * 99
    glyphs = codecs.GLYPHS.reshape(94, 64)
    mse = ((glyphs[:, None] - glyphs[None]) ** 2).mean(-1) + np.eye(94)
    if hamming.min() < 1 or mse.min() <= 0:
        raise AssertionError("Prototypes are not distinct")
    return {"passed": True, "round_trips": counts, "min_bgv_hamming": float(hamming.min()),
            "min_cgge_mse": float(mse.min()), "font_sha256": FONT_SHA256}


def parameter_gate():
    from .models import make_model, parameter_count
    reg = registration()
    expected, actual = {}, {}
    for pipeline, spec in reg["pipelines"].items():
        counts = reg["models"][spec["model"]]["parameters"]
        expected[pipeline] = counts[spec["representation"]] if spec["model"] == "G3-U" else counts[spec["source"]]
        actual[pipeline] = parameter_count(make_model(pipeline, "A-impl", 0))
    expected["P-DISC:D1-T-L"] = reg["models"]["D1-T-L"]["parameters"]
    actual["P-DISC:D1-T-L"] = parameter_count(make_model("P-DISC", "A-impl", 0, "D1-T-L"))
    if actual != expected:
        raise AssertionError(f"Parameter counts differ: {actual} != {expected}")
    return {"passed": True, "parameters": actual}


def sampler_gate(s, budget=None):
    from .models import (candidate_keys, make_model, sample_images, sample_images_reference,
                         sample_tokens, sample_tokens_reference)
    pool = np.array(data.synthetic_split()["acceptance"], dtype=np.int32)
    result = {"discrete": {}, "gaussian": {}}
    for pipeline in ("P-DISC", "R-DISC"):
        model = make_model(pipeline, "A-impl", 0)
        n = s["d1_candidates"]
        labels = np.resize(pool, n)
        keys = candidate_keys(("A-impl", pipeline, "sampler", 0), np.arange(n))
        reference = sample_tokens_reference(model, labels, keys)
        for batch in sorted({64, min(1024, n)}):
            actual = np.concatenate([sample_tokens(model, labels[o:o + batch], keys[o:o + batch]) for o in range(0, n, batch)])
            if not np.array_equal(actual, reference):
                raise AssertionError(f"D1 sampler changed with batch={batch}: {pipeline}")
        if not all(data.valid(m, model.src) for m in data.decode(reference, model.src)):
            raise AssertionError(f"D1 sampler produced invalid candidates: {pipeline}")
        result["discrete"][pipeline] = {"candidates": n, "batches": [1, 64, min(1024, n)], "bitwise": True}
    for pipeline in ("P-G-BGV", "P-G-CGGE", "R-G-BGV"):
        model = make_model(pipeline, "A-impl", 0)
        n, m = s["g3_candidates"], s["g3_reference"]
        labels = np.resize(pool, n)
        keys = candidate_keys(("A-impl", pipeline, "sampler", 0), np.arange(n))
        rows = {}
        for steps in s["g3_steps"]:
            first = None
            for batch in s["g3_batches"]:
                if budget:
                    budget.check()
                pieces = [sample_images(model, labels[o:o + batch], keys[o:o + batch], steps) for o in range(0, n, batch)]
                images, lengths = np.concatenate([x[0] for x in pieces]), np.concatenate([x[1] for x in pieces])
                if first is None:
                    first = images, lengths
                elif not (np.array_equal(images, first[0]) and np.array_equal(lengths, first[1])):
                    raise AssertionError(f"G3 batch invariance failed: {pipeline}, steps={steps}, batch={batch}")
            reference, reference_lengths = sample_images_reference(model, labels[:m], keys[:m], steps)
            error = float(np.max(np.abs(reference - first[0][:m])))
            a = codecs.decode(reference, reference_lengths, model.representation, model.src)[0]
            b = codecs.decode(first[0][:m], first[1][:m], model.representation, model.src)[0]
            agree = sum(x == y for x, y in zip(a, b))
            if not np.array_equal(reference_lengths, first[1][:m]) or error > 1e-3 or agree < m - m // 256:
                raise AssertionError(f"G3 scalar reference differs: {pipeline}, steps={steps}, error={error}, agree={agree}/{m}")
            rows[str(steps)] = {"candidates": n, "batches": s["g3_batches"], "bitwise": True,
                                "reference_candidates": m, "reference_agree": agree, "reference_max_abs_error": error}
        result["gaussian"][pipeline] = rows
    model = make_model("P-DISC", "A-impl", 0, "D1-T-L")
    keys = candidate_keys(("A-impl", "D1-T-L", "sampler", 0), np.arange(8))
    if not np.array_equal(sample_tokens(model, pool[:8], keys), sample_tokens_reference(model, pool[:8], keys)):
        raise AssertionError("D1-T-L sampler differs from its scalar reference")
    result["scale"] = {"candidates": 8, "bitwise": True}
    return {"passed": True, **result}


class _Interrupt:
    """Raise at the first budget check after a checkpoint pointer exists."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def check(self):
        if (self.folder / "checkpoint.json").exists():
            raise InterruptedError("fixture interruption after the first checkpoint")


def _same_parameters(a, b):
    from mlx.utils import tree_flatten
    for (ka, va), (kb, vb) in zip(tree_flatten(a.parameters()), tree_flatten(b.parameters()), strict=True):
        if ka != kb or not np.array_equal(np.asarray(va), np.asarray(vb)):
            raise AssertionError(f"Training parameters differ: {ka}")


def training_gate(folder):
    from . import runtime
    folder = Path(folder)
    groups = data.synthetic_split()
    families = []
    for pipeline, architecture in (("P-DISC", None), ("R-DISC", None), ("P-G-BGV", None), ("P-G-CGGE", None), ("P-DISC", "D1-T-L")):
        name = pipeline if architecture is None else f"{pipeline}-{architecture}"
        kwargs = dict(pipeline=pipeline, stage="A-impl", seed_id=0, groups=groups, task="synthetic", batch_size=8,
                      checkpoint_every=2, diagnostic_pairs=2, architecture=architecture)
        direct = runtime.train(folder / name / "direct", updates=8, **kwargs)
        try:
            runtime.train(folder / name / "resumed", updates=4, budget=_Interrupt(folder / name / "resumed"), **kwargs)
            raise AssertionError("Fixture interruption did not fire")
        except InterruptedError:
            pass
        resumed = runtime.train(folder / name / "resumed", updates=4, **kwargs)
        four = runtime.train(folder / name / "four", updates=4, **kwargs)
        _same_parameters(resumed, four)
        continued = runtime.train(folder / name / "continued", updates=8, resume_from=folder / name / "four", **kwargs)
        _same_parameters(direct, continued)
        if read_json(folder / name / "resumed" / "attempt.json")["retries"] != 1:
            raise AssertionError("Resume was not counted as the single retry")
        if read_json(folder / name / "continued" / "contract.json")["resume_from"]["update"] != 4:
            raise AssertionError("Continuation contract lost its parent")
        families.append(name)
    shuffled = folder / "P-DISC-shuffled"
    runtime.train(shuffled, "P-DISC", "A-impl", 0, groups, updates=2, method="Shuffled", task="synthetic",
                  batch_size=8, checkpoint_every=2, diagnostic_pairs=2)
    main_contract = read_json(folder / "P-DISC" / "direct" / "contract.json")
    shuffled_contract = read_json(shuffled / "contract.json")
    if main_contract["namespaces"] != shuffled_contract["namespaces"]:
        raise AssertionError("Main and Shuffled must share weights, messages and corruption")
    segment = np.load(shuffled / "segments" / "seg-00000002.npz")
    identity = hashlib.sha256(np.arange(8).astype("<u4").tobytes()).digest()
    if any(bytes(row) == identity for row in segment["perm_sha256"]):
        raise AssertionError("Shuffled permutation was not recorded")
    bgv_contract = read_json(folder / "P-G-BGV" / "direct" / "contract.json")
    if (bgv_contract["namespaces"]["fresh"] != main_contract["namespaces"]["fresh"]
            or bgv_contract["namespaces"]["weights"] == main_contract["namespaces"]["weights"]):
        raise AssertionError("Pipelines of one source must share messages but not weights")
    return {"passed": True, "families": families, "resume_bitwise": True, "continuation_bitwise": True,
            "main_shuffled_shared": True, "source_stream_shared": True}


def stream_gate():
    for src in data.SOURCES:
        for window, rung in (("W1", 64), ("W3", 64), ("W4", 64), ("W1", 4)):
            groups = data.split(window, rung)
            a = data.fresh_batch(("A-impl", src, 0), 3, 256, groups["train"], window=window, rung=rung)
            b = data.fresh_batch(("A-impl", src, 0), 3, 256, groups["train"], window=window, rung=rung)
            if not all(np.array_equal(x, y) for x, y in zip(a, b, strict=True)):
                raise AssertionError(f"Stream is not deterministic: {src}, {window}, r={rung}")
            hashes = data.hash_batch(a[0], a[1], rung, window)
            if not (np.array_equal(hashes, a[2]) and np.isin(hashes, groups["train"]).all()):
                raise AssertionError(f"Train-group rejection failed: {src}, {window}, r={rung}")
        payload, lengths, labels, _ = data.fresh_batch(("A-impl", src, 0), 3, 256, data.synthetic_split()["train"], task="synthetic")
        if [data.synthetic_label(m, src) for m in data.messages(payload, lengths)] != labels.tolist():
            raise AssertionError(f"Synthetic prefix does not encode the label: {src}")
    for forbidden in (lambda: data.split("W2", 64),
                      lambda: data.fresh_batch(("A-impl", "P", 0), 0, 8, [0], window="W2")):
        try:
            forbidden()
        except ValueError:
            continue
        raise AssertionError("Forbidden W2 was accepted")
    for size in (2, 3, 100, 4096):
        donors = data.derangement(("A-impl", size), size)
        if np.any(donors == np.arange(size)) or len(set(donors.tolist())) != size:
            raise AssertionError("MC donors are not a derangement")
    namespace = ("A-impl", "prior")
    for src in data.SOURCES:
        whole = data.prior_candidates(namespace, np.arange(20), src)
        pieces = [data.prior_candidates(namespace, np.arange(o, min(o + 3, 20)), src) for o in range(0, 20, 3)]
        if not all(np.array_equal(whole[i], np.concatenate([x[i] for x in pieces])) for i in (0, 1)):
            raise AssertionError("Prior candidates depend on batch composition")
    return {"passed": True, "windows": ["W1", "W3", "W4"], "rungs": [64, 4], "w2_refused": True}


def clp_gate():
    from .models import candidate_keys, clp_pairs, make_model
    from .runtime import encoded_batch
    for pipeline, spec in registration()["pipelines"].items():
        model = make_model(pipeline, "A-impl", 0)
        payload, lengths, labels, _ = data.fresh_batch(("clp", "A-impl", "fixture", spec["source"], 0), 0, 8,
                                                       data.synthetic_split()["validation"], task="synthetic")
        clean = encoded_batch(payload, lengths, spec["representation"], spec["source"])
        keys = candidate_keys(("clp-corruption", "A-impl", "fixture", pipeline, 0), np.arange(4))
        d = clp_pairs(model, clean, lengths, labels, keys)
        opposite = clp_pairs(model, clean, lengths, labels.reshape(-1, 2)[:, ::-1].reshape(-1), keys)
        if not np.array_equal(d, -opposite):
            raise AssertionError(f"CLP is not antisymmetric: {pipeline}")
    return {"passed": True, "pipelines": list(registration()["pipelines"])}


def ledger_gate(folder):
    from . import runtime
    folder = Path(folder)
    if runtime.RECORD.itemsize != registration()["ledger"]["record_bytes"]:
        raise AssertionError("Ledger record size changed")
    targets = np.array(data.split("W1", 64)["validation"][:128])
    options = dict(stage="A-impl", source="P", method="Random", seed_id=0, batch=256)
    summary, result = runtime.evaluate_block(folder / "direct", 1, targets, **options)

    class Stop:
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls == 12:
                raise InterruptedError("partial block")

    try:
        runtime.evaluate_block(folder / "resumed", 1, targets, budget=Stop(), **options)
        raise AssertionError("Ledger interruption did not fire")
    except InterruptedError:
        pass
    resumed, _ = runtime.evaluate_block(folder / "resumed", 1, targets, **options)
    if not np.array_equal(summary, resumed) or (folder / "direct/block-1.bin").read_bytes() != (folder / "resumed/block-1.bin").read_bytes():
        raise AssertionError("Resumed ledger differs from the uninterrupted ledger")
    records = np.memmap(folder / "resumed/block-1.bin", dtype=runtime.RECORD, mode="r+")
    records["flags"][0] ^= 2
    records.flush()
    try:
        runtime.verify_ledger(folder / "resumed/block-1.bin", targets, result["metadata"])
        raise AssertionError("Tampered hit flag was accepted")
    except ValueError:
        pass
    records["flags"][0] ^= 2
    namespace = result["metadata"]["namespace"]
    indices = np.arange(len(records), dtype=np.uint64)
    chosen = indices[data.key_words(("regen-audit", namespace, 1), indices)[:, 1] % 100 == 0]
    index = int(chosen[0])
    records["payload"][index, 0] = 33 if records["payload"][index, 0] != 33 else 34
    records.flush()

    def generate(values):
        p, n = data.prior_candidates(namespace, values, "P")
        return data.messages(p, n), None, None

    try:
        runtime.regeneration_audit(folder / "resumed/block-1.bin", namespace, 1, 0, len(records), generate)
        raise AssertionError("Tampered payload passed the regeneration audit")
    except ValueError:
        pass
    meta = {"source": "P", "task": "synthetic", "rung": 64, "window": "W1", "method": "Random"}
    ledger = runtime.Ledger(folder / "fixture", 1, meta, [0, 0], batch=3, k=3)
    ledger.append(0, [b"000!", b"000!", None])
    ledger.append(3, [b"FFFF", b"000!", b"~~~~"])
    ledger.commit()
    ledger.close()
    trials, metrics = runtime.verify_ledger(ledger.path, [0, 0], meta, k=3)
    if metrics["rows"] != 6 or metrics["hits"] != 3 or metrics["duplicates"] != 1 or trials["first"].tolist() != [0, 1]:
        raise AssertionError("Invalid or duplicate candidates did not consume attempts")
    training = runtime.message_digests([b"000!"])
    ledger = runtime.Ledger(folder / "match", 1, meta, [0], batch=1, k=1, training=training)
    ledger.append(0, [b"000!"])
    ledger.commit()
    ledger.close()
    try:
        runtime.verify_ledger(ledger.path, [0], meta, k=1, training=training)
        raise AssertionError("A successful training message was accepted")
    except ValueError:
        pass
    return {"passed": True, "record_bytes": runtime.RECORD.itemsize, "resume_bitwise": True,
            "flag_tamper_detected": True, "payload_tamper_detected": True, "attempt_accounting": True}


def _planted_preimage(target):
    for draw in range(10000):
        payload, lengths = data.source(data.rng("A-impl", "planted-table", draw), 4096, "P")
        hits = np.flatnonzero(data.hash_batch(payload, lengths, 64, "W3") == target)
        if len(hits):
            i = int(hits[0])
            return payload[i, :lengths[i]].tobytes()
    raise AssertionError("No planted preimage found for the validation target")


def planted_gate(folder, trials, *, quick=False, budget=None):
    """Full ledger path on one W3 validation target; the study test pool is never used."""
    from . import runtime
    folder = Path(folder)
    target = int(data.split("W3", 64)["validation"][0])
    preimage = _planted_preimage(target)
    targets = np.full(trials, target, dtype=np.int32)
    planted = int(round(st.DELTA / (1 - st.P0) * trials))
    outcomes = {"Random": [], "Main": [], "Shuffled": []}
    for seed_id in range(3):
        chosen = set(data.rng("A-impl", "planted-trials", seed_id).choice(trials, planted, replace=False).tolist())
        for method, prior in (("Random", "coupled"), ("Main", "coupled"), ("Shuffled", "independent")):
            namespace = ("A-impl", "planted", prior, seed_id)
            meta = {"protocol": PROTOCOL, "stage": "A-impl", "source": "P", "method": method, "task": "md5",
                    "window": "W3", "rung": 64, "representation": None, "fixture": "planted-lift"}
            ledger = runtime.Ledger(folder / f"{method}-{seed_id}", 1, meta, targets, batch=1024)
            for offset in range(ledger.position, trials * 100, 1024):
                if budget:
                    budget.check()
                indices = np.arange(offset, offset + 1024, dtype=np.uint64)
                payload, lengths = data.prior_candidates(namespace, indices, "P")
                messages = data.messages(payload, lengths)
                if method == "Main":
                    for i, index in enumerate(indices.tolist()):
                        if index % 100 == 0 and index // 100 in chosen:
                            messages[i] = preimage
                ledger.append(offset, messages)
            ledger.commit()
            ledger.close()
            summary, _ = runtime.verify_ledger(ledger.path, targets, meta, budget=budget)
            outcomes[method].append(summary["at100"].astype(int))
    arrays = {name: np.stack(values) for name, values in outcomes.items()}
    block = trials // registration()["stage_c"]["looks"]
    decisions = {}
    for case, main in (("plus_zero", arrays["Random"]), ("plus_delta", arrays["Main"])):
        for look in range(1, registration()["stage_c"]["looks"] + 1):
            t = block * look
            result = st.stage_c({"fixture": {"Main": main[:, :t], "Random": arrays["Random"][:, :t],
                                             "Shuffled": arrays["Shuffled"][:, :t]}}, look)
            if result["action"] == "stop":
                break
        decisions[case] = {"decision": result["pipelines"]["fixture"]["decision"], "look": look}
    expected = {"plus_zero": "REJECTED_BOUNDED", "plus_delta": "POSITIVE"}
    matched = all(decisions[k]["decision"] == v for k, v in expected.items())
    if not quick and not matched:
        raise AssertionError(f"Planted-lift fixture failed: {decisions}")
    return {"passed": matched or quick, "decisions_checked": not quick, "decisions": decisions,
            "trials": trials, "planted_trials_per_seed": planted, "validation_target": target, "window": "W3"}


def implementation_gate(root, quick=False, budget=None):
    s = sizes(quick)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v6-checks-", dir=root) as temporary:
        folder = Path(temporary)
        gates = {"G1": hash_gate(s["hash_messages"]), "G2": codec_gate(), "G3": parameter_gate(),
                 "G4": sampler_gate(s, budget), "G5": training_gate(folder / "training"), "G6": stream_gate(),
                 "G7": clp_gate(), "G8": ledger_gate(folder / "ledger"), "G9": st.calibrate(s["calibration"]),
                 "G10": planted_gate(folder / "planted", s["planted_trials"], quick=quick, budget=budget)}
    # 200-repetition calibration is a path check; its CP bounds can miss the criteria by chance.
    quick_paths = (all(g["passed"] for name, g in gates.items() if name != "G9")
                   and gates["G9"]["scope"] == "regression-only" and len(gates["G9"]["scenarios"]) == 4)
    return {"passed": not quick and all(g["passed"] for g in gates.values()), "quick": quick,
            "quick_paths_passed": quick_paths if quick else None,
            "certifies_study": not quick, "gates": gates, "sizes": s,
            "study_md5_condition_pairs": 0, "hash_calls_scope": "implementation-fixtures-only"}
