# Canonical Research Plan: Diffusion-Based Hash Preimage Candidate Generation

**Status: RESEARCH PLAN DOCUMENTED — SPECIFICATION FREEZE STILL REQUIRED**

This is the authoritative scientific protocol for subsequent implementation, validation, execution, analysis, interpretation, and follow-up work. It establishes the initial canonical specification under the supplied research requirements. It is a prospective protocol, not a report of completed experiments. No scientific gate is certified by this document, and no measurements or experimental results are asserted.

**Authority and repository context.** Inspection found Python source, tests, examples, a dependency lock, and pre-existing planning documents, including a file at this path. Those files are implementation context, not authority to change this specification. This document replaces the contents at the canonical path; other files are left unchanged. Conflicting directions in `RESEARCH_PLAN_POC.md`, `RESEARCH_PLAN_GAUSSIAN_DISCRETE.md`, `NEW_EXPERIMENT_PLAN.md`, examples, or code do not override this protocol. Appendix A records observed discrepancies and required future work. No training, experiment execution, or implementation changes are part of this documentation task.

**Normative terms.** MUST and MUST NOT are requirements. A **FIXED** item has a defined protocol value or invariant; this does not mean its implementation has passed validation. **TBD** identifies an unresolved choice. **BLOCKING** identifies an unresolved decision or missing evidence that prevents the affected activity. **OPTIONAL** identifies work outside the primary matrix, disabled unless explicitly activated. Every execution-critical TBD means **TBD — MUST FREEZE BEFORE EXECUTION**, unless a later deadline is explicitly stated. A partial freeze is not permission to execute the affected scientific experiment.

## 1. Executive Summary

Stage I tests whether a diffusion model conditioned only on a truncated MD5 target generates valid preimages more often than both source-prior random search and an otherwise matched model trained with shuffled conditions. Evaluation uses fixed source distributions, disjoint digest groups, unique digest targets, and fixed candidate budgets. The five primary pipelines span Gaussian BGV, Gaussian CGGE, and categorical masked diffusion; they yield 15 primary model/digest-size settings. Candidate budgets are prefixes of one ordered stream, not separately trained settings.

The current target is seed-replicated, truncated hash-conditioned candidate-generation evidence. Computational advantage, digest scaling, full-MD5 evidence, and comparison with cryptanalytic attacks require separate later studies. The split design, model configurations, CGGE specification, controls, statistical decision criteria, and resource limits still require freeze. Length-leakage safety requires implementation evidence before any primary hash experiment.

## 2. Research Motivation

A generator may improve over naive candidate generation merely by approximating a source prior. That observation alone does not establish use of the hash condition. A matched shuffled-condition training control is therefore essential. Small truncated digests allow inexpensive checks of wiring and candidate accounting, but also create saturation, collision groups, and limited independent evaluation units. The protocol distinguishes these effects from generalization to unseen hash conditions.

The long-term question is whether conditional generative models can produce useful statistical or computational advantages in preimage search. This protocol begins with a restricted, falsifiable question; it does not presume that any advantage exists or extrapolates to a cryptographic attack.

## 3. Research Questions and Hypotheses

For a fixed source, pipeline, digest size, seed, target set, and candidate budget, let $P_{\mathrm{main}}$, $P_{\mathrm{random}}$, and $P_{\mathrm{shuffled}}$ denote target-level preimage-success probabilities under their declared sampling procedures. The two directional comparisons are

$$
P_{\mathrm{main}}>P_{\mathrm{random}},
\qquad
P_{\mathrm{main}}>P_{\mathrm{shuffled}}.
$$

Each comparison has a null of no positive advantage. Evidence for hash-conditioned candidate advantage requires both comparisons for the **same** setting. A gain against random alone can reflect source-prior modeling. A gain against shuffled alone can coexist with a model that is worse than random search.

Secondary questions concern Gaussian BGV versus Gaussian CGGE and Gaussian versus Discrete pipelines. The former is a controlled representation ablation; the latter is an end-to-end pipeline comparison. Hypothesis membership, directions, significance level, confidence level, and the precise G3 rule remain to be frozen. Observed positive signs alone are not corrected statistical significance.

## 4. Scope and Explicit Non-Claims

Stage I concerns source-distribution-restricted, truncated-MD5 preimages with unknown source length at generation time. It MUST NOT be described as full MD5 inversion, arbitrary-target MD5 inversion, a cryptographic break, asymptotic complexity reduction, lower complexity than state-of-the-art attacks, superiority to the best-known MD5 preimage attack, SHA-256 inversion, or full-digest computational advantage.

Candidate-generation advantage means greater verified success at equal candidate opportunities. Computational advantage concerns explicitly measured and matched work. General cryptanalytic advantage additionally requires an appropriate attack definition and comparison. None implies the next automatically. Collision finding, exact source recovery, and preimage finding are different tasks.

## 5. Terminology and Formal Definitions

A **source message** $x$ is a raw byte string from the declared distribution. Its truncated digest $y=H_q(x)$ is a **target**. Multiple source messages may share one target. A **candidate** $\hat{x}$ is one attempted generated byte string or a recorded invalid decode. An **attempt** is one generation opportunity, regardless of validity, duplication, or success.

A **setting** is one primary pipeline and one primary $q$ value. A **run** additionally identifies method, seed, frozen configuration, dataset, and checkpoint. Main and shuffled training are separate runs within a setting. The statistical unit is the unique digest target, not a message, candidate, token, image pixel, or seed-target row.

Let $\operatorname{Valid}(\hat{x})$ require a successful frozen deterministic decode and membership in the declared source domain: length from 4 through 31 and allowed payload symbols. Validity never requires matching the hidden source length or recovering the original source message. The verifier hashes decoded payload bytes only, excluding representation headers, EOS, PAD, and image pixels.

## 6. Overall Research Program

| Stage | Question | Entry condition | Maximum intended evidence |
|---|---|---|---|
| I — Hash-Conditioned Candidate Generation PoC | Does hash conditioning improve candidate success against both controls? | Specification freeze and the blocking gates below | F2, seed-replicated candidate evidence |
| II — Restricted-Domain Compute Efficiency | Does the gain survive explicit work accounting and compute matching? | Recommended: P2 and CA0; new preregistration and uncontaminated data | C1 |
| III — Digest-Difficulty Scaling | How does measured work change with digest difficulty? | Stage-II success under its frozen compute criterion; new scaling protocol | C2 |
| IV — Full-MD5 / SOTA Comparison | Is full-digest work lower than a comparable best published attack? | Full-digest study justified, matched attack definition, new literature review and preregistration | C3, then S1 only if supported |

Stages II–IV are future research, not authorized executable extensions of Stage I.

## 7. Stage I — Hash-Conditioned Candidate Generation PoC

### 7.1 Source Distributions

Printable ASCII excludes space:

$$
\mathcal{A}_{P}=\{\mathtt{0x21},\ldots,\mathtt{0x7E}\},
\qquad |\mathcal{A}_{P}|=94.
$$

Random Bytes uses all byte values:

$$
\mathcal{A}_{R}=\{\mathtt{0x00},\ldots,\mathtt{0xFF}\},
\qquad |\mathcal{A}_{R}|=256.
$$

For both sources, draw length uniformly and payload symbols independently and uniformly given that length:

$$
L\sim U\{4,\ldots,31\},\qquad L_{\max}=31,
\qquad P(x)=\frac{1}{28}A^{-|x|}.
$$

The last expression applies to domain-valid $x$ for alphabet size $A$; probability is zero outside that domain. Uniform over lengths is not uniform over all byte strings. No language corpus, frequency-weighted characters, Unicode normalization, space insertion, or altered length prior may replace this source. Raw byte serialization must preserve NUL and high-byte values.

### 7.2 Hash Definition and Canonical Truncation

Use standard MD5 over raw payload bytes as defined by [RFC 1321](https://www.rfc-editor.org/rfc/rfc1321). The reference defines the digest algorithm and its test vectors; its historical security conjectures are not claims of this protocol.

$$
H_q(x)=\operatorname{Trunc}_q(H_{\mathrm{MD5}}(x)),
\qquad q\in\{8,12,16\}.
$$

Canonical truncation is **the first $q$ bits of the standard serialized digest, with the most significant bit first within each digest byte**. Preserve the 16 digest bytes in the order returned by a conforming MD5 digest API; do not reverse bytes or reinterpret internal MD5 words. Equivalently,

$$
v_q(x)=\left\lfloor\frac{\operatorname{Int}_{\mathrm{big}}(H_{\mathrm{MD5}}(x))}{2^{128-q}}\right\rfloor.
$$

The condition is the exactly $q$-bit, zero-padded binary expansion of $v_q(x)$. Log targets as lowercase, zero-padded hexadecimal with exactly $q/4$ digits for the declared sizes. Thus 12 bits means the first three hexadecimal digits, including a possible leading zero. Byte storage may carry unused low bits only if set to zero and ignored by all encoders. The full digest must not reach the generation interface.

The independent canonical verifier must check MD5 test vectors and prefix extraction, including the half-byte boundary at 12 bits and leading zeros. Hand-written hash traces and model-produced digest values cannot be the sole verification authority.

Optional $q=20$ is disabled by default. Activation requires a dated, pre-outcome amendment defining targets, controls, resource limits, hypothesis families, and claim scope; it remains a separately labeled stretch study. It does not silently expand the 15-setting primary matrix. Full MD5 has $q=128$ and is excluded from Stage I.

### 7.3 Dataset Construction

For each of the two source distributions, the stipulated source-message counts are:

| Provisional split | Source messages before digest-group handling |
|---|---:|
| Train | 10,000 |
| Validation | 1,000 |
| Test | 1,000 |

These are source-message counts, not unique target counts. Unless a frozen construction explicitly revises their semantics, they mean 12,000 initial source draws with the listed provisional allocations. They do not promise those exact final counts after enforcing whole-group separation. Report requested draws, actual draws, raw duplicates, rejected or unused draws, retained messages, and final counts separately.

**Construction algorithm: BLOCKING — TBD — MUST FREEZE BEFORE EXECUTION.** The prompt does not fix whether grouping changes final counts or whether a new quota-conditioned construction yields exact retained counts. Do not silently take either path. In particular, sampling independently into three fixed splits will almost certainly reuse short digest conditions and is not an admissible final split.

The frozen algorithm must provide deterministic steps for: PRNG identity/version; dataset and split seeds; length and symbol draws; raw duplicate treatment; stable ordering; digest computation; group ownership; quotas or post-group counts; tie-breaking; stopping limits; unsuccessful-construction handling; representative selection; and manifest serialization. Raw duplicate draws must be assigned to a single owner or removed under the frozen rule; rejection sampling changes the retained distribution and must be disclosed. Baseline candidate sampling always remains independent source-prior sampling with replacement.

Audit prior access before reusing repository data or seeds. Because an implementation and experiment archive already exist, no stored holdout is presumed uncontaminated. Record its exposure history or construct a new holdout under the frozen protocol; never reuse an exposed holdout simply because this document is new.

Two admissible design classes require an explicit decision:

- **Fixed initial corpus, whole-group allocation:** draw the prescribed initial corpus, resolve raw duplicates deterministically, and allocate complete groups. Freeze allocation weights and tie-breaking. Report resulting counts; never split a group or claim exact final message quotas that were not achieved.
- **Exact retained quotas:** explicitly redefine the counts as retained messages. Assign digest ownership without using model outcomes; draw source messages until each quota is met under immutable ownership, a fixed draw cap, and a frozen rejection rule. Log conditioning, surplus, duplicates, and every discarded draw count. Failure to fill a split is a construction failure, not permission to reseed repeatedly until a favorable corpus appears.

Neither class is adopted merely by being described here. The freeze record must select one and contain complete pseudocode/configuration. Record the effect of split conditioning on the realized length and symbol distributions. Do not describe retained samples as independent draws from the original prior without qualification. No balancing by observed model success is allowed.

### 7.4 Split Independence and Nested Truncation

For every distinct pair of splits $a,b\in\{\mathrm{train},\mathrm{validation},\mathrm{test}\}$, require

$$
X_a\cap X_b=\varnothing,
\qquad Y_a^q\cap Y_b^q=\varnothing
\quad\text{for each evaluated }q.
$$

Audit raw bytes and digest groups, not record IDs. Zero forbidden overlap is mandatory. Audit again immediately before evaluation and bind results to dataset hashes.

**Cross-digest-size design: BLOCKING — TBD — MUST FREEZE BEFORE EXECUTION.** Select and document one of the following, consistently with the construction algorithm:

- **Universal split:** assign entire 8-bit prefix groups to splits once per source. Because longer prefixes share their first eight bits, this also separates 12- and 16-bit groups. Report that held-out longer targets occupy entirely unseen 8-bit prefix regions, a stronger and different generalization task than a split at each larger size. Split counts and effective target counts may be limited by only 256 possible groups.
- **Separate $q$-specific splits:** construct each split at its own digest size. Within each size, all pipelines for the same source share the same corpus, split, and targets. Cross-size overlap is possible; treat settings as separate studies with separately initialized models and no cross-size transfer, checkpoint reuse, test-informed tuning, or pooled claims of independent observations. Document reduced comparability across sizes. The ownership audit must address any shared base corpus and access to cross-size target messages.

Models are trained separately for each primary setting; no joint cross-size model is part of the primary matrix. Neither a universal split nor separate splits make cross-size outcomes independent. A seed-0 test result at 8 bits may not guide a later confirmatory 12- or 16-bit design. All scientific choices for the primary matrix must be frozen before any primary test is opened. Engineering debugging must use separate fixtures/development data; exposure followed by redesign requires uncontaminated evaluation and an amended preregistration.

### 7.5 Evaluation Unit and Target Population

Evaluate unique digest targets once per method and seed. For each final split and source/digest setting, report: source-message count, unique digest count, and evaluated target count. The default evaluation scope is all unique test targets; any resource-limited subsampling rule is an execution-critical freeze decision and must be outcome-blind and shared by all methods.

$$
N_{\mathrm{test}}^q=|Y_{\mathrm{test}}^q|\le 2^q,
\qquad 2^8=256.
$$

With nonempty disjoint training and validation digest sets, fewer than 256 values remain for 8-bit testing. A 1,000-message test corpus cannot supply 1,000 independent 8-bit observations. Never duplicate a target's success label for each colliding message.

The estimand averages success equally over the retained unique test targets. It is conditional on the frozen source construction, split rule, checkpoint-selection process, and declared randomness, not an arbitrary uniform 128-bit target experiment. Deduplicating digests changes the weighting relative to drawing source messages. Report this explicitly. Select an outcome-independent source representative for diagnostics using a frozen stable rule; keep the complete collision group for audit. Representatives and other source metadata are evaluator-only.

### 7.6 BGV — Byte Glyph Visualization

BGV is the general byte-to-image representation for both sources. Geometry is fixed:

| Property | Definition |
|---|---|
| Logical slots | 32: slot 0 is length; slots 1–31 are payload |
| Slot grid | Four rows and eight columns, row-major |
| Byte glyph | Eight bits in a $2\times4$ logical glyph, row-major and most significant bit first |
| Bit expansion | Each logical bit fills a $4\times4$ pixel block |
| Cell dimensions | Height 8, width 16 |
| Image dimensions | Height 32, width 128 |
| Channels | Glyph/value, then validity |
| Tensor shape | $[2,32,128]$ |

Slot 0 encodes the byte value of payload length. Slots 1 through $L$ encode payload bytes. Unused glyph slots are zero before normalization. The validity channel fills the header and used payload cells with one and unused cells with zero. Consequently, a valid payload `0x00` has a zero glyph and validity one; padding has validity zero. This mapping follows reusable BGV conventions inspected in the repository.

The clean encode/decode contract is

$$
D_{\mathrm{BGV}}(E_{\mathrm{BGV}}(x))=x.
$$

Use decoded, generated header and validity information only. Reject inconsistent header/mask, out-of-range length, nonfinite or wrong-shaped output, source-alphabet violations, and padding violations under the frozen strict decoder. Block aggregation, bit and validity thresholds, threshold tie rules, output clipping, and exact padding rejection rules must be frozen; existing defaults do not automatically establish their scientific values. Normalization and inverse normalization must agree with §7.9 and preserve exact clean round trips.

**Length analysis.** Clean training data necessarily contain length. The primary generation process must jointly generate the complete glyph/header and validity channels from full-shape noise. Neither channel may be initialized from a target, protected from corruption, clamped to its true value, supplied as a mask, or repaired using the hidden length. Training denoising inputs are corrupted training examples; evaluating denoising of a corrupted test image is reconstruction, not primary generation. Decoder length must be a model output. End-to-end compliance remains **BLOCKING pending the frozen sampler/decoder and leakage tests**, even though these restrictions are fixed.

### 7.7 CGGE — Character Glyph Grid Encoding

CGGE applies only to Printable ASCII. BGV maps a character to its byte bits; CGGE maps it to a deterministic visual glyph. Its role is the Gaussian representation ablation `P-G-BGV` versus `P-G-CGGE`.

Supported alphabet is the same 94 characters. The following are **TBD — MUST FREEZE BEFORE EXECUTION**: font identity and font-file/table checksum and version; renderer/library/version; font size; glyph dimensions; cell dimensions; image/grid dimensions and channels; character-to-cell placement; length representation; padding representation; normalization; decoder; classification distance/rule; thresholds and ties; invalid-decode policy; and visually ambiguous-glyph treatment. An existing embedded glyph table is an implementation candidate, not automatic scientific selection.

Rendering and decoding must be deterministic across the declared environment. Verify all supported glyphs are distinguishable under the frozen clean decoder. A tie or indistinguishable pair must not be silently resolved by source knowledge. Freeze deterministic ambiguity rejection or redesign before test exposure; a codec that cannot recover every clean character fails G1-A.

$$
D_{\mathrm{CGGE}}(E_{\mathrm{CGGE}}(x))=x
$$

must hold for 100% of the preregistered correctness corpus. That corpus includes every character in valid-length messages, all allowed lengths, repetitions, mixed case, digits, punctuation, and ambiguous-looking pairs. The corpus seed and exact construction remain to be frozen.

**Length analysis.** Any length header, validity channel, blank padding, image extent, spatial mask, or auxiliary metadata must be generated jointly, from a fixed public shape independent of the target message. Neither target glyphs nor a target-derived padding image may enter sampling. Classification may inspect only the generated image and frozen codec parameters. This design is **BLOCKING** until the representation and sampler are fixed and audited. Do not assume BGV's safety transfers automatically to CGGE.

### 7.8 Discrete Token Representation

The sequence length is fixed at

$$
L_{\max}+1=31+1=32.
$$

Use the existing payload-first token convention as the canonical byte mapping:

| Source | Payload token IDs | EOS | PAD | MASK | Vocabulary size |
|---|---|---:|---:|---:|---:|
| Printable | Byte value minus 33, from 0 to 93 | 94 | 95 | 96 | $94+3=97$ |
| Random Bytes | Byte value, from 0 to 255 | 256 | 257 | 258 | $256+3=259$ |

The clean sequence is

$$
[x_1,\ldots,x_L,\mathrm{EOS},\mathrm{PAD},\ldots,\mathrm{PAD}].
$$

Here the payload symbols denote their corresponding token IDs. PAD is a separate categorical state: $\mathrm{PAD}\ne\mathtt{0x00}$. Random-Bytes token zero is a real payload NUL. Decode exactly one EOS at an allowed length, payload states before it, and only PAD after it. Reject remaining MASK, unknown or noninteger states, a wrong sequence length, missing/multiple EOS, invalid payload, or inconsistent suffix. Do not truncate to the first EOS while ignoring malformed output, repair suffixes, or resample invalid outputs. At length 31 there is one EOS and no PAD. Require exact TokenCodec round trips on the correctness corpus.

**Length analysis.** Corrupt all 32 positions, including EOS and PAD, under the same rule. Sampling starts with 32 MASK tokens and a fixed public attention domain. No true EOS position, PAD locations, length, or padding attention mask is supplied. The model generates EOS and PAD. Sequence grammar is checked after generation; hidden target length never guides repair or decoding. End-to-end wiring evidence is required before the leakage gate can pass.

### 7.9 Gaussian Diffusion

Let $z_0=\mathcal{N}(E(x))$ be the normalized clean image and $\epsilon\sim\mathcal{N}(0,I)$ full-tensor Gaussian noise. The forward-process contract is

$$
z_t=\sqrt{\bar{\alpha}_t}\,z_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon.
$$

Noise must apply to all channels and spatial locations, including header, validity, and padded regions. Any loss weighting or mask must be declared and must not create a path carrying the true target structure into inference. Generation starts with independent Gaussian noise over the full fixed tensor shape and receives only the allowed condition. A noisy test encoding, even at a large noise level, is not a valid starting state.

The following are **TBD — MUST FREEZE BEFORE EXECUTION**: architecture and parameter count; conditioning encoder; clean normalization and inverse; prediction target/parameterization; training-timestep distribution and count; noise schedule and terminal behavior; loss and weighting; sampler; sampling steps; clipping; optimizer; learning-rate search; regularization; updates, batch size, and numerical precision. Epsilon prediction, clean-sample prediction, a U-Net width, or a common schedule default is not fixed by this document. Existing code is a candidate implementation only.

The frozen forward process, training objective, prediction parameterization, and reverse sampler must be mathematically consistent. If the terminal training distribution only approximates independent noise, document the approximation and check its effect in the positive control. The sampler may not compensate using target-derived data.

### 7.10 Discrete Diffusion

The primary training corruption uses continuous time:

$$
t\sim U(0,1),\qquad m_j\mid t\sim\operatorname{Bernoulli}(t),
\qquad
z_{t,j}=\begin{cases}
\mathrm{MASK},&m_j=1,\\
z_{0,j},&m_j=0.
\end{cases}
$$

Masks are independent across positions conditional on $t$. Payload, EOS, and PAD are all eligible. Train the conditional model to predict original clean tokens. Exact loss reduction, which positions contribute, weighting, and behavior when no positions are masked are freeze decisions; none may require a true target mask at inference.

Architecture, time representation/embedding, objective details, reverse-time grid, transition probabilities, remasking, temperature, sampling steps, optimizer, learning-rate search, and training budget are **TBD — MUST FREEZE BEFORE EXECUTION**. The sampler begins from all MASK and uses only the declared condition plus its independent randomness. Finite reverse integration is compatible with continuous-time training if explicitly specified; replacing continuous training-time draws by a finite training schedule is a protocol change and must be resolved before testing. No implicit reinterpretation is permitted.

### 7.11 Hash Conditioning and Information Boundary

All learned pipelines receive the same information: the canonical $q$-bit vector from §7.2. Source identity, algorithm MD5, and $q$ are public run constants. A representation-specific encoder may transform these bits, including deterministic zero-padding, but must not receive additional target data. Its architecture is a freeze item.

The target-specific generator input contract is the bit vector only. Randomness is assigned through an opaque scheduling interface; record IDs and ordering are not model features. The generator must not accept the source message, source-message ID, semantically informative target ID, true length, true EOS, PAD positions, target padding mask, hidden suffix, full MD5 digest, or metadata revealing structure. Full digests and source representatives may be retained in evaluator-only records, never in the model-facing batch.

The scientific target is

$$
p(x\mid H_q(x)),
$$

not

$$
p(x\mid H_q(x),L).
$$

Known-length inference would require a separately labeled, separately preregistered experiment with matched length-aware controls. It is outside the primary matrix.

### 7.12 Leakage Prevention and Required Evidence

The information boundary is a blocking part of G0, with representation/sampler checks also required for G1-B and G2. Before scientific execution, the future implementation must provide:

1. A traced generation call graph showing only the allowed condition, public run constants, checkpoint, and independent randomness reach the sampler and decoder.
2. A mutation test: with digest bits, checkpoint, and RNG state held fixed, change or remove evaluator-only source message, length, full-digest suffix, source ID, EOS/PAD mask, and representative. Generated candidates must be unchanged, or access must be rejected before sampling.
3. Full-shape initialization/corruption checks for every representation. BGV header/validity, CGGE length/padding structures, and all token positions must not preserve target-specific information.
4. Tests of leading-zero and partial-byte truncation, and assurance that hidden suffix changes do not change conditioning.
5. Tests that batch collation, sorting, attention masks, cache keys, filenames, and RNG assignment do not encode true length or other hidden target data into model inputs.
6. A verifier-only path for source recovery diagnostics, separated from candidate generation. Evaluation results must not feed back into retries, candidate ranking, or stopping.

These are required future checks, not completed validations. A codec round trip or a successful reconstruction does not certify leakage safety. Unresolved BGV or CGGE structure handling, or unresolved EOS/PAD handling, blocks the affected primary experiment.

### 7.13 Experimental Pipeline Matrix

Exactly these five primary pipelines are permitted:

| ID | Source | Generative model | Representation |
|---|---|---|---|
| P-G-BGV | Printable ASCII | Gaussian Diffusion | BGV |
| P-G-CGGE | Printable ASCII | Gaussian Diffusion | CGGE |
| P-DISC | Printable ASCII | Discrete Diffusion | Token |
| R-G-BGV | Random Bytes | Gaussian Diffusion | BGV |
| R-DISC | Random Bytes | Discrete Diffusion | Token |

Each is evaluated at $q\in\{8,12,16\}$, giving $5\times3=15$ primary model/$q$ settings. Candidate budgets $K\in\{1,10,100\}$ are evaluation prefixes, not trained settings. Shuffled runs, positive controls, and seed replication add runs but not primary pipeline IDs. CGGE is not a Random-Bytes pipeline. Direct Bits, SHA-256, known-length branches, deterministic predictors, and optional larger digests are excluded from this primary matrix.

### 7.14 Source-Prior Random Baseline

For every setting and seed, sample each candidate independently with replacement from the exact source prior in §7.1. Draw a fresh length for each attempt; do not use the target's source length, a learned length distribution, digest-group ownership, or a rejection filter. Use the same targets, digest size, budgets, domain validity, verifier, and outcome calculation as Main. The baseline's direct byte generation is valid without an image decoder; this difference is part of end-to-end comparison.

A single immutable baseline stream may be reused across pipelines sharing source, target set, digest size, and seed if that reuse policy is frozen. Generate target-specific random streams; do not reuse one candidate stream across all targets and then pretend target outcomes are independent. Controls must be regenerated or reused by policy, not according to which gives Main a better effect.

For a fixed target $y$, define source digest mass

$$
p_y=\sum_{x:H_q(x)=y}P(x).
$$

Independent source-prior attempts then give

$$
P(\mathrm{success\ within\ }K\mid y)=1-(1-p_y)^K.
$$

Under the idealized equal-mass assumption $p_y=2^{-q}$, this becomes

$$
P(\mathrm{success\ within\ }K)=1-(1-2^{-q})^K.
$$

The equal-mass assumption is an approximation for the actual finite source domain and retained targets. Average the target-specific expression over the actual target population when relevant; do not substitute the idealized formula for measured paired baseline outcomes. No numerical baseline performance is asserted here.

### 7.15 Shuffled-Condition Negative Control

For each learned pipeline, train a separate otherwise matched model. Main training pairs are $x_i\leftrightarrow H_q(x_i)$; shuffled pairs are

$$
x_i\leftrightarrow H_q(x_{\pi(i)}).
$$

Donors must come from the same training split, source, and digest size. Validation donors, if used by the frozen control-selection policy, must be confined to validation. No test source or condition may be a training donor. Do not stratify by hidden target length as part of primary generation.

**Permutation algorithm, shuffle seed, fixed-versus-resampled permutation policy, collision/fixed-point handling, and shuffled validation/checkpoint selection are TBD.** Freeze a reproducible permutation procedure and record the donor map/checksum. A permutation without index fixed points can still pair equal digests; audit accidental correct-condition matches. If using digest derangement, specify feasibility checks and abort behavior. Repeated draws until a favorable control is obtained are forbidden. A fixed finite random permutation can itself be memorized; report this residual limitation.

Keep architecture, representation, optimizer, updates, hyperparameter search resources and selection policy, sampler, and checkpoint-selection policy matched. Any unavoidable difference must be registered. Both Main and shuffled models are evaluated on the **same actual target $y_i$ supplied as the inference condition**, and the verifier checks against $y_i$. Shuffling test conditions at inference answers a different question and cannot replace this negative control. Hyperparameter selection for the shuffled model must follow the frozen matched policy rather than an outcome-driven attempt to weaken it.

### 7.16 Conditional-Generation Positive Controls

Require an actual learned conditional-generation positive control for Gaussian-BGV, Gaussian-CGGE, and Discrete-Token, covering both source domains where applicable. It must exercise the real condition encoder, model, corruption, training objective, reverse sampler, decoder, and verifier interface, starting from the same target-independent initial state used for hash generation.

The task must have a known, learnable relation between a permitted synthetic condition and the desired output, with held-out examples and a check that using the condition matters. A reversible synthetic task is an admissible candidate, not an already selected task. If a task uses a different condition width, freeze its adapter and demonstrate that it checks the actual hash-condition path sufficiently; a shortcut that simply copies the source into output is inadequate.

Tasks, dataset sizes, seeds, learning budget, metrics, thresholds, control comparisons, and pass logic are **TBD — MUST FREEZE BEFORE EXECUTION**. Codec correctness alone, an oracle that returns the original bytes, and denoising a partially visible test example are insufficient. A failed positive control yields **HASH EXPERIMENT FOR THAT PIPELINE = BLOCKED**. Such failure diagnoses the pipeline; it is not evidence that the scientific hypothesis is false.

### 7.17 Candidate Budget and Attempt Ledger

For each target, method, and seed, generate one ordered stream of exactly 100 opportunities. Evaluate its first candidate, first 10, and all 100:

$$
K\in\{1,10,100\},\qquad
\mathrm{Success@1}\le\mathrm{Success@10}\le\mathrm{Success@100}.
$$

Invalid candidates and duplicates consume attempts. Continue the predeclared stream after a success. Do not regenerate for validity, uniqueness, or success; do not generate a larger pool and report only a ranked subset. One stochastic reverse trajectory is one candidate opportunity; internal denoising steps contribute work, not extra candidate opportunities unless they are decoded or searched as additional candidates. No hidden verifier-guided repair, rejection, beam search, or hash-based reranking is part of the primary sampler.

Write one ledger row per opportunity with target, method, seed, position, candidate bytes or invalid marker, decoder reason, validity, verifier outcome, and timing/work references. Duplicates are verified and counted again; no result-dependent caching may change accounting. Invalid outputs remain failures with zero or explicitly recorded actual verifier calls, since no valid byte string may exist to hash.

A crash is not permission to discard unfavorable completed attempts. Resume from a frozen checkpoint/RNG state without replacing completed ledger rows; if exact continuation is impossible, mark the run incomplete and follow the preregistered rerun policy. A missing stream cannot be relabeled as a complete zero-success run. Candidate-budget fairness is not compute fairness.

### 7.18 Primary Outcome

For each unique digest target $i$, define

$$
S_i(K)=\mathbf{1}\left[\exists j\le K:
\operatorname{Valid}(\hat{x}_{ij})\land H_q(\hat{x}_{ij})=y_i\right].
$$

Aggregate with equal target weight:

$$
\mathrm{PreimageSuccess@K}=\frac{1}{N}\sum_{i=1}^{N}S_i(K).
$$

A candidate may differ from every stored source representative and still be a valid preimage. The canonical independent MD5/truncation verifier decides success. Image or glyph similarity, BER, CER, token accuracy, or reconstruction loss must never substitute for it. The verifier must operate on the declared truncated target, never require a hidden suffix match, and never enforce true source length.

### 7.19 Secondary Diagnostics

Record these at every declared prefix, with denominators and missingness:

| Diagnostic | Required interpretation |
|---|---|
| ExactSourceRecovery@K | Fraction of unique targets for which the chosen outcome-independent source representative is recovered; identify representative-selection rule |
| ValidDecodeRate | Valid, source-domain candidates divided by all attempted candidates |
| Duplicate rate | Repeated decoded byte strings after their first occurrence within a target stream, divided by all attempts; also report valid-attempt denominator and raw counts |
| Generated-length distribution | Length histogram among valid candidates plus invalid/malformed length counts separately |
| BER/CER | Only under a frozen alignment/reference rule, with denominators and treatment of invalid output; a different valid preimage need not resemble the reference |
| BGV diagnostics | Header/mask consistency, padding violations, clean and generated glyph decoding behavior |
| CGGE diagnostics | Character/glyph accuracy, ambiguity and threshold rejection rates |
| Discrete diagnostics | EOS placement, PAD consistency, remaining MASK, token accuracy on declared reference tasks |

Exact recovery of any message in a collision group may be an additional labeled diagnostic; it must not silently replace representative recovery. Reference-based diagnostics are evaluator-only, conditioned on a nonunique source representative, and not evidence of primary success. Freeze detailed metric definitions before test exposure.

### 7.20 Gaussian Representation Ablation

Compare `P-G-BGV` against `P-G-CGGE`, holding source data, splits, unique targets, digest size, permitted condition, Gaussian model family, training budget, candidate budget, sampling budget, and verifier constant as far as possible. Freeze which budget is matched: updates, examples, parameters, NFE, or another declared quantity. These are not automatically equal compute.

If image dimensions necessitate architecture or parameter-count changes, record them and qualify the interpretation. Differences in validity rates and decoder rejection belong in the analysis. Report this as a **Gaussian representation ablation under the declared matching constraints**, not a comparison of all encodings. Family B governs confirmatory statistical claims.

### 7.21 Gaussian versus Discrete Pipeline Comparison

`P-G-BGV` versus `P-DISC` and `R-G-BGV` versus `R-DISC` vary representation, state space, architecture, corruption, objective, sampler, and decoder simultaneously. They are **end-to-end pipeline comparisons**, not pure representation experiments. Comparisons of `P-G-CGGE` and `P-DISC` are exploratory unless separately preregistered. These comparisons do not acquire confirmatory status by borrowing Family A or Family B results.

### 7.22 Statistical Analysis

Use same-target paired binary outcomes at a fixed setting, seed, and budget. Let $M_i$ be Main success and $B_i$ the relevant control success. Define

$$
\Delta_K=\frac{1}{N}\sum_{i=1}^{N}(M_i-B_i),
\qquad n_{10}=\#\{i:M_i=1,B_i=0\},
\qquad n_{01}=\#\{i:M_i=0,B_i=1\}.
$$

Also report $n_{00}$ and $n_{11}$ and confirm their sum is $N$. For preregistered directional primary comparisons, use **one-sided exact McNemar**, conditional on $n=n_{10}+n_{01}$:

$$
p_{\mathrm{raw}}=\sum_{r=n_{10}}^{n}\binom{n}{r}2^{-n}.
$$

If $n=0$, set the p-value to one. Use the exact binomial tail, not an asymptotic chi-square approximation or unregistered mid-p variant. The binomial-test interface and its one-sided alternative are documented in the [SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html); that reference does not require adding SciPy as a dependency.

The exact-test interpretation assumes independent target pairs with the null discordant direction probability one-half. A weaker null about average effects across heterogeneous targets does not by itself guarantee that condition. Declare the inferential population and this assumption; do not equate digest uniqueness with guaranteed statistical independence. Shared checkpoints, constrained splits, finite target populations, and heterogeneous target difficulty limit generalization. Avoid shared cross-target sampling randomness and do not pool related digest sizes or candidates as extra observations. Freeze the handling of any unavoidable dependence before execution; unresolved incompatibility with the paired analysis blocks confirmatory claims.

Compute a **paired target-level bootstrap interval using 10,000 resamples**. Each resample draws $N$ target indices with replacement and retains each selected target's complete paired outcome; recompute $\Delta_K$. If jointly reporting several budgets or controls, preserve the corresponding target row together. Never bootstrap candidates, pixels, or the two methods independently. Shared-index resampling is the paired bootstrap described in the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

Confidence level, interval method, quantile convention, handling of degeneracy, and bootstrap RNG/seed are **TBD — MUST FREEZE BEFORE EXECUTION**. A degenerate empirical interval when every observed difference is identical does not establish absence of uncertainty outside the observed support. Label intervals as marginal unless simultaneous coverage has been separately specified; Holm-corrected p-values do not make ordinary intervals simultaneous.

Analyze seeds separately. Seed replication uses the same frozen targets and dataset unless an independently preregistered data-replication study is added. It tests algorithmic randomness on those targets, not independent population replication. Do not stack the three seeds to claim $3N$ independent targets. Report effect size, confidence interval, full paired counts, raw p-value, adjusted p-value, family membership, and gate outcome. No p-value-only reporting and no post-hoc target exclusions.

### 7.23 Multiple Comparisons and Adaptive Eligibility

Keep the two scientific families separate and apply **Holm correction within each family**:

| Family | Scientific question | Members to register before tests |
|---|---|---|
| A — Hash-conditioned evidence | Does Main beat both controls? | Main > Source-Prior Random and Main > Shuffled for each included pipeline, digest size, budget, and inferential seed/stage |
| B — Gaussian representation ablation | Does representation change success? | P-G-BGV versus P-G-CGGE for the registered digest/budget/seed scope and direction |

For sorted raw p-values $p_{(1)}\le\cdots\le p_{(m)}$, the adjusted value at rank $j$ is

$$
\widetilde p_{(j)}=\min\left(1,\max_{1\le r\le j}\{(m-r+1)p_{(r)}\}\right).
$$

Map results back to their original hypothesis IDs. Freeze stable tie ordering. Holm handles dependence among valid component p-values; it does not fix invalid component tests, target leakage, or outcome-driven family membership.

**Exact membership and directions: BLOCKING — TBD — MUST FREEZE BEFORE TEST EVALUATION.** Register a literal hypothesis inventory including seed scope, budgets, digest sizes, directional versus two-sided ablation test, significance level, missing-run handling, and family size. For orientation only, including all 15 settings, all three budgets, and both controls at seed 0 would give 90 Family A hypotheses; restricting to the ten 12-/16-bit settings at budget 100 would give 20. These are alternative counts, not selected analysis plans. Family B must declare its own scope and direction; do not choose the direction after seeing the result.

P1-based seed replication creates selection. Freeze the complete eligible universe and replication algorithm **before seed-0 testing**. Retain all seed-0 results; never correct only the favorable selected subset. Specify either a valid preregistered hierarchical procedure or a full prospective family across possible replication tests, with a conservative predefined treatment of unperformed tests. Do not silently perform separate per-seed correction and claim it controls a combined multi-seed search. No pooling across seeds or cherry-picking the strongest seed. Unresolved selective-inference handling blocks confirmatory G3 claims; P1 signs remain descriptive eligibility only.

### 7.24 Scientific Gates

All gate evidence must name scope, artifact hashes, code revision, criteria, status, and reason. Gate states are not inferred from the presence of files. Missing evidence is NOT RUN or BLOCKED, never PASS.

Gate evidence is configuration-specific. Any change to a validated codec, condition path, architecture, objective, sampler, or relevant training/selection procedure invalidates affected earlier gate evidence. Repeat the affected checks and positive controls before hash testing; a development configuration's pass does not transfer automatically to the selected configuration.

| Gate | Required evidence | Failure consequence |
|---|---|---|
| G0 — Data and evaluation independence | Zero raw-message and forbidden digest overlap; deterministic construction; nested-size/access policy; untouched tests; permitted information boundary | Downstream scientific evaluation blocked for affected data/settings |
| G1-A — Codec correctness | $D(E(x))=x$ on 100% of the preregistered correctness corpus for BGV, CGGE, and TokenCodec in their applicable domains | Affected representation blocked |
| G1-B — Conditional-generation positive control | Actual learned pipeline passes its frozen task and threshold, with full generation and no target leakage | Affected pipeline's hash experiment blocked |
| G2 — Candidate and comparison fairness | Same target set and prefixes; actual attempts equal declared budget; invalids/duplicates counted; canonical verifier; no success-based regeneration or selection | Affected comparisons invalid and downstream claims blocked |
| G3 — Statistical evidence | Preregistered paired test, effect direction, intervals, and correct Holm families satisfy frozen criterion against both controls | No corrected statistical-advantage claim for that setting |
| G4 — Seed reproducibility | Eligible settings complete seeds 0, 1, 2 with positive effects against both controls in every seed, plus any frozen stronger criterion | No P2 for that setting |

G1 means both G1-A and G1-B. The correctness corpus must cover all legal lengths, all payload symbols in valid messages, extrema, repetitions, mixed symbols, NUL/high bytes where applicable, padding/header boundaries, and invalid-structure rejection fixtures. Exact corpus construction and seed are TBD. Clean-codec correctness at 100% is fixed; a positive-control success threshold is not.

**G3 decision thresholds: TBD — MUST FREEZE BEFORE TEST EVALUATION.** Define significance level, confidence level/method, required lower-bound rule, any minimum practically relevant effect, and whether G3 is an added P1/P2 eligibility requirement. No conventional threshold is adopted by implication. Without this freeze, G3 cannot be judged. A statistically valid negative result is different from a failed measurement gate.

G4's minimum is positive effect direction for both controls in all three seeds. Requiring each seed's confidence lower bound above zero is an optional stronger criterion that must be formally selected before seed-0 results. Report the criterion actually used.

### 7.25 PoC Evidence Levels and the Role of 8 Bits

**P0 — Pipeline feasible** requires

$$
G0\land G1\land G2.
$$

**P1 — Initial hash-conditioned signal** requires P0 and, for the same setting at $q=12$ or $q=16$, $K=100$, seed 0,

$$
\Delta_{\mathrm{random}}>0,
\qquad \Delta_{\mathrm{shuffled}}>0.
$$

These signs are the minimum eligibility criterion. Report tests and intervals regardless. The exact P1-to-P2 selection rule and any added G3 requirement are TBD before testing. No outcome-driven top-setting limit or extra threshold may be added later.

**P2 — Seed-replicated feasibility** requires preregistered P1 eligibility, P0 on all required runs, and G4 across seeds $0,1,2$ against both controls. Report all statistical uncertainty. Under the minimum sign-based definition, P2 can occur without corrected statistical significance; it must then be described as replicated positive directions, not established statistical superiority. A stronger statistical P2 rule is allowed only if frozen in advance.

The 8-bit settings primarily check pipeline wiring, conditioning, verifier behavior, accounting, and basic learning signals. They do not alone establish P1. Target saturation and limited target count must be reported. An 8-bit test is still a test: debugging after opening it cannot silently alter later confirmatory settings.

### 7.26 Compute Accounting

Collect work telemetry from the beginning. **CA0 — Compute Accounting Complete** means every preregistered mandatory field is present and auditable. It does not mean advantage.

| Field | Recording rule |
|---|---|
| Candidate opportunities and MD5 calls | Actual counts, including duplicates; distinguish invalid attempts and verifier calls |
| NFE and sampling steps | Define NFE as denoiser evaluations; record batched calls and per-candidate evaluations, including guidance if used |
| Training | Updates, examples processed, wall-clock, model/optimizer settings; account for Main, shuffled, tuning, positive controls separately |
| Inference and verification | Separate sampling and verification wall-clock, with synchronization and timing boundary specified |
| CPU/GPU time | Measure where supported; record measurement method and unavailable status |
| Memory | Peak VRAM; peak RAM where practical; record instrumentation |
| Model and arithmetic | Batch size, numerical precision, parameter count, FLOPs estimate where practical with estimation method |
| Storage | Checkpoint size, raw candidate/result sizes, retained artifact totals |
| Environment | Hardware/accelerator, software, dependency lock, code revision |

Mandatory-versus-optional telemetry and unavailable-field policy are freeze items. An unavailable field cannot be silently recorded as zero. If a mandatory field cannot be collected, CA0 fails. Include actual work for all 100 attempts, even if success occurs early. NFE, hash calls, wall time, and FLOPs remain distinct units in Stage I; no neural forward pass is equated with an MD5 evaluation.

### 7.27 Execution Procedure and Freeze Discipline

Follow this order; an affected blocking failure stops downstream work. Unaffected pipelines may proceed only under the frozen partial-failure and family-membership policy, never by deleting failures from the study.

| Phase | Action | Required output |
|---|---|---|
| 0 | Specification freeze, including selection rules, hypotheses, resources, seeds, leakage design | Dated immutable protocol/configuration, hypothesis inventory, artifact identities |
| 1 | Dataset and split construction/validation | G0 report with overlap, counts, provenance and access audit |
| 2 | BGV, CGGE, TokenCodec validation | G1-A reports and correctness-corpus manifest |
| 3 | Pipeline-specific learned conditional positive controls | G1-B reports under frozen tasks and thresholds |
| 4 | Candidate accounting and paired-comparison validation on fixtures | G2 reports, verifier fixtures, complete attempt ledgers |
| 5 | 8-bit sanity experiments | Frozen seed-0 results without redesign from primary tests |
| 6 | 12- and 16-bit seed-0 feasibility experiments | Main, random, shuffled outcomes at all three prefixes |
| 7 | P1 eligibility assessment | Complete eligible/ineligible inventory under the predeclared rule |
| 8 | Eligible seed-1 and seed-2 replication | Same dataset/targets and fixed procedures; no seed hunting |
| 9 | P2 and CA0 assessment, G3 analysis | Full per-seed effects, uncertainty, correction and telemetry |
| 10 | Final Stage-I report | Exactly one exit state per declared scope, all failures and non-claims |

Scientific constants, search spaces, tuning budgets, selection algorithms, and test-access policy are frozen in Phase 0. Training and validation may then select hyperparameters/checkpoints only inside that frozen policy. Before Phase 5, freeze selected configurations and checkpoints or the deterministic checkpoint-selection rule for later seed-specific training. Later test results may not modify any of them. In functional terms,

$$
\mathrm{Train}\longrightarrow\mathrm{Validation\ selection}
\longrightarrow\mathrm{Freeze}\longrightarrow\mathrm{Test}.
$$

For replicated seeds, retrain under the same frozen configuration and checkpoint policy; do not select seed-specific favorable variants. Freeze data, split, model, training-noise, sampling, shuffle, random-baseline, representative, and bootstrap seed namespaces and derivation. Model seeds 0, 1, 2 are fixed; the other seed values and PRNG versions are TBD. Do not couple random draws to test successes or unrecorded execution order.

Validation protocol, checkpoint metric/direction/ties, evaluation frequency, early stopping, learning-rate search, and tuning resource limits are TBD. The test set cannot choose architecture, learning rate, sampler, checkpoint, stopping rule, CGGE design, representation, threshold, hypothesis family, or optional extension. If a test motivates a change, create a revised preregistration and uncontaminated evaluation scope; a new registration alone does not make already exposed data untouched. Preserve the original run as exploratory or invalid as appropriate, rather than overwriting history.

### 7.28 Stage-I Exit Criteria

Each completed reporting scope receives exactly one state:

| State | Definition |
|---|---|
| A — Gate Failure | Required measurement/independence gate failed, required run or accounting incomplete, or protocol violation prevents interpretable assessment |
| B — P0 Only | Feasible and complete evaluation, but no P1 under the frozen rule |
| C — P1 but not P2 | Initial eligible signal, but the completed replication does not meet the frozen P2 criterion |
| D — P2 + CA0 | Seed-replicated feasibility and complete mandatory compute accounting |

Use A for P2 with incomplete CA0 so that accounting incompleteness is not mislabeled as D or as absence of replication. A failed G3 superiority threshold in otherwise valid data is not automatically A; report its statistical outcome alongside B, C, or D according to the frozen P1/P2 definition. A failure of G4's effect criterion after completed valid replication yields C, not an infrastructure gate failure. The present document is pending freeze, not a completed Stage-I outcome.

The scope and deterministic rule for rolling per-setting states into one overall Stage-I exit state are **TBD — MUST FREEZE BEFORE EXECUTION**. Specify required settings, permitted partial failures, and aggregation; never call all pipelines successful because one is. Report every setting's state even if an overall label is used. Recommended Stage-II entry is

$$
P2\land CA0.
$$

This is permission to investigate compute efficiency, not evidence it will succeed.

### 7.29 Specification Freeze Checklist for Executable Stage I

This checklist closes the executable Stage-I specification. Section 14 defines how it is approved; Section 15 groups the open decisions. A fixed invariant with a separate blocking evidence row is intentional.

| Item | State | Required disposition |
|---|---|---|
| Source alphabets | FIXED | 94 printable non-space ASCII symbols; all 256 bytes |
| Length prior and maximum | FIXED | Uniform 4–31, independent uniform symbols; maximum 31 |
| Primary digest sizes | FIXED | MD5, 8/12/16 bits |
| Optional 20 bits | OPTIONAL | Disabled; separate prospective activation and analysis |
| Candidate budgets/stream | FIXED | 1/10/100 prefixes of 100, invalids and duplicates counted |
| Dataset size targets | FIXED | 10,000/1,000/1,000 per source, pre-group semantics unless explicitly revised at freeze |
| Dataset-construction algorithm | BLOCKING | Choose and fully specify grouping/count semantics, duplicate policy, PRNG, draw caps, failure behavior |
| Split algorithm | BLOCKING | Whole digest ownership, raw-byte separation, weights/order/ties and realized counts |
| Nested-size handling | BLOCKING | Universal 8-bit grouping or isolated size-specific datasets; access and comparability policy |
| Target selection and diagnostic representatives | TBD | All unique targets unless registered subsampling; deterministic representative rule |
| BGV geometry and clean mapping | FIXED | Two-channel 32-by-128 image and row-major byte glyph/header/validity semantics |
| BGV complete decoder | TBD | Aggregation/thresholds/ties, strict padding, clipping and normalization |
| BGV leakage handling | BLOCKING | Frozen full-noise sampler plus end-to-end boundary evidence |
| CGGE supported alphabet | FIXED | Printable only |
| CGGE complete codec | BLOCKING | Dimensions, layout/channels, length/padding, normalization, classifier, thresholds, invalids, ambiguity |
| CGGE font | TBD | Identity, file/table checksum/version, size |
| CGGE renderer | TBD | Library/version and deterministic rasterization |
| CGGE leakage handling | BLOCKING | Joint generation of all length-bearing structures; fixed target-independent shape and tests |
| TokenCodec | FIXED | Payload-first IDs, distinct EOS/PAD/MASK, strict 32-position grammar |
| EOS/PAD leakage handling | BLOCKING | All-position corruption/all-MASK generation rule fixed; integrated implementation proof required |
| Hash-condition information/bit order | FIXED | Exactly declared prefix, MSB-first serialized digest; no target metadata |
| Hash-conditioning encoder | TBD | Architecture/adapters, shape and equivalent information audit |
| Gaussian architecture | TBD | Family, width/depth, parameter budget and conditioning |
| Gaussian normalization/objective/prediction target | TBD | Clean scaling/inverse, target, loss weighting and reduction |
| Gaussian schedule | TBD | Training-time distribution, timesteps, noise values, terminal approximation |
| Gaussian sampler | TBD | Reverse equations, clipping/guidance if any |
| Gaussian sampling steps | TBD | Fixed number and NFE accounting |
| Discrete architecture | TBD | Sequence model, embeddings and conditioning |
| Continuous-time corruption | FIXED | Uniform continuous time; independent masking probability equal to time for all positions |
| Discrete time/objective implementation | TBD | Time representation, loss/reduction, zero-mask behavior |
| Discrete reverse process | TBD | Grid, transition probabilities, remasking |
| Discrete sampler | TBD | Sampling steps, temperature, categorical procedure |
| Optimizers and searches | TBD | Per-family optimizer, learning rates/search, regularization and equal control policy |
| Training budgets | TBD | Updates/examples/batch, tuning budget, stop/failure/resume rules |
| Validation procedure | TBD | Metrics, frequency, target use, search selection and tie rules |
| Checkpoint-selection rule | TBD | Metric, direction, ties, seed-specific use of same rule |
| Correctness corpus | TBD | Exact fixtures, coverage, seed and manifest; clean success threshold fixed at 100% |
| Positive-control tasks | BLOCKING | Actual learned tasks/adapters for all applicable families/sources |
| Positive-control thresholds | BLOCKING | Metrics, budgets, held-out scope and pass rules |
| Shuffled training control | TBD | Permutation/seed/collision policy, validation use; true target at inference fixed |
| Source-prior baseline | FIXED | Exact original prior with replacement, target-specific independent streams |
| Baseline stream reuse | TBD | Explicit reuse across comparable pipelines and seed derivation |
| Statistical families | BLOCKING | Exact A/B inventories, directions, seed/budget scope and adaptive replication handling |
| Paired inference/resampling | FIXED | Exact one-sided McNemar for directional tests; paired target bootstrap with 10,000 resamples; Holm per family |
| Interval/population/dependence details | TBD | Confidence level, method/quantiles, degenerate case, conditional inferential scope |
| G3 criterion | BLOCKING | Significance, interval and effect requirements, relationship to P1/P2 |
| P1-to-P2 eligibility | BLOCKING | Minimum signs fixed; full selection rule, scope and stronger criteria unresolved |
| Model replication seeds | FIXED | 0, 1, 2; no seed selection |
| Seed policy | TBD | Data/split/training/sampling/shuffle/baseline/bootstrap namespaces, values and PRNG versions |
| Ablation matching | TBD | Registered budget matching and unavoidable model differences |
| Secondary metric conventions | TBD | Alignment, denominators, reference selection and invalid handling |
| Compute telemetry | TBD | Instrumentation, mandatory fields, unavailable policy, timing/NFE definitions |
| Hardware | TBD | CPU, accelerator model/count, execution environment |
| Numerical precision | TBD | Training/sampling arithmetic and determinism policy |
| Wall-clock budget | TBD | Per-run/study limits, interruption policy |
| Storage budget | TBD | Candidate/checkpoint/result limits and retention |
| Overall exit-state rule | TBD | Scope, required settings, partial-failure treatment and deterministic roll-up |
| Reproducibility bundle | TBD | Final schema/locations, immutable freeze and linkage checks |

## 8. Stage II — Restricted-Domain Compute Efficiency

Stage II requires a new preregistration, uncontaminated evaluation data, and an explicit compute-matching methodology. Its primary question is whether the Stage-I candidate advantage persists when work is accounted for and matched. Define

$$
\mathrm{PreimageSuccess@Compute}(B)
$$

using a frozen work budget $B$, stopping rule, and full accounting boundary. Match source domain, target definition, hardware policy, preprocessing, memory, and verification requirements. Include compute-matched source-prior search and any other registered competitive baselines; retain candidate-matched reporting separately.

Separate $C_{\mathrm{train}}$, $C_{\mathrm{inference}}$, and $C_{\mathrm{verification}}$. If the latter two are study totals, use

$$
W_{\mathrm{total}}=C_{\mathrm{train}}+C_{\mathrm{inference}}+C_{\mathrm{verification}}.
$$

For $N$ targets, define per-target online costs explicitly and use

$$
W_{\mathrm{amortized,target}}=\frac{C_{\mathrm{train}}}{N}
+C_{\mathrm{inference,target}}+C_{\mathrm{verification,target}}.
$$

If shorter notation omits the target subscript, it must state that online terms are per-target, not totals. Include tuning/preprocessing/control costs in declared accounts; any exclusions must be transparent and cannot disappear into amortization. Report how many targets can legitimately share training and preprocessing, and both total and amortized results.

A neural forward pass is not one MD5 compression-function evaluation. Freeze a defensible common work metric and implementation/hardware efficiency assumptions before claims. Wall time alone is hardware-specific; hash counts alone omit neural work. Stage-II success and the criterion for entry to Stage III must be preregistered here before execution of that future study, including reproducibility, uncertainty, resource limits, and matched-work evidence. Stage I does not preselect their numerical values.

## 9. Stage III — Digest-Difficulty Scaling

Only methods that meet the frozen Stage-II compute-evidence criterion may proceed. Preregister new evaluation data, budgets, stopping/success criteria, model-selection rules, and treatment of censored or failed runs. Suggested progression is

$$
q=20\longrightarrow24\longrightarrow32,
$$

and only if later justified,

$$
q=64\longrightarrow128.
$$

Measure empirical work $W(q)$ to reach a preregistered success criterion and analyze $\log_2 W(q)$ with uncertainty. Include training, online work, verification, memory, and any required changes in domain or architecture. Report runs that never reach the criterion as censored or failed, not as absent data. Do not fit only successful sizes or silently extrapolate over a changed task.

Any extrapolation to 128 bits is an **empirical projection**, not a **proof of asymptotic complexity**. A favorable slope on a small truncated range does not establish full-digest feasibility. Stage-IV entry requires a separately justified full-digest design and resources, not merely a fitted line.

## 10. Stage IV — Full-MD5 / SOTA Comparison

At Stage IV's start, conduct a fresh literature review of the then-current best published full-MD5 preimage attacks. Do not hard-code a current work factor or permanently designate a particular paper as SOTA in this plan.

A valid comparison must match the attack problem as closely as possible: arbitrary 128-bit target, full MD5, preimage definition, allowed messages/output constraints, online computation, preprocessing, memory, training cost, amortization, success probability, and work unit. A source-derived restricted target study is not automatically comparable with arbitrary-target preimage cryptanalysis. Distinguish full-hash attacks from reduced-round or compression-function results and from collisions.

The final research question is

$$
W_{\mathrm{Diffusion}}(128)<W_{\mathrm{SOTA}}(128)?
$$

Only evidence under the matched full-digest attack definition can answer it. Stage I cannot; neither Stage II nor Stage III alone can. If work is estimated rather than executed, report the empirical and analytical components separately with explicit uncertainty and assumptions.

## 11. Overall Evidence Hierarchy

| Level | Meaning |
|---|---|
| F0 | Pipeline feasibility |
| F1 | Truncated hash-conditioned candidate advantage |
| F2 | Seed-replicated candidate advantage |
| C1 | Restricted-domain compute-matched advantage |
| C2 | Favorable empirical digest-difficulty scaling |
| C3 | Full-digest computational evidence |
| S1 | Lower work factor than SOTA under a matched attack definition |

Stage I targets F2. P0 supports F0. P1 is an initial signal and P2 is a seed-replication designation; promotion to a statistically supported F1/F2 advantage additionally requires the frozen G3 evidence. Sign-only P2 must not be advertised as significant advantage. CA0 records accounting completeness, not C1.

$$
F2\ne C1\ne C2\ne C3\ne S1.
$$

Likewise P2 is neither computational advantage, full-MD5 evidence, nor SOTA improvement. Use the strongest level actually supported, not the level the program hopes to reach.

## 12. Threats to Validity

Documentation is not mitigation evidence. Every residual risk below must be addressed in the report even after a relevant gate passes.

| Threat | Risk | Required mitigation / gate | Residual limitation |
|---|---|---|---|
| Small digest sizes | Easy truncated tasks may reward artifacts absent at larger sizes | Separate 8-bit sanity from P1; Stage-III preregistration | Strong small-size results may not scale |
| Digest-target saturation | Many messages collapse to few targets; large budgets saturate success | Unique-target counts, complete paired tables, G0/G2 | Limited power and ceiling effects remain |
| Finite-source effects | Digest masses and retained distributions need not be uniform | Exact source definition, construction logs, measured random baseline | Conditional domain differs from arbitrary targets |
| Source-prior exploitation | Better source modeling can beat random without hash use | Main versus matched shuffled control, G3/G4 | Imperfect shuffled training can still confound attribution |
| Target-length leakage | Header or metadata exposes privileged structure | Full-generation boundary/mutation checks, G0/G1-B/G2 | Unchecked call paths remain a risk until audited |
| EOS/PAD leakage | True suffix masks reveal length | All-position masking and all-MASK start; no target attention mask | Learning length from permitted hash bits is possible and allowed |
| Image-representation leakage | Noisy target, fixed validity, padding or dimensions encode length | Fixed public shape, full noise, joint generation, G1-B/G2 | CGGE design unresolved until freeze |
| Train/test digest overlap | Memorized conditions inflate apparent generalization | Whole-group split plus raw-byte audit, G0 | Unseen prefixes still share bit structure |
| Nested-size leakage | One size's train/test data inform another size | Frozen universal or isolated policy; no transfer/test tuning, G0 | Cross-size comparisons remain dependent |
| Hyperparameter overfitting | Extensive validation searches favor chance variants | Fixed search budgets, selection rules, matched controls | Finite validation sets still permit overfitting |
| Test adaptation | Repeated redesign uses the holdout as training feedback | Immutable pre-test freeze; new holdout/preregistration after changes, G0 | Access history must be documented |
| Duplicate candidates | Deduplication grants extra effective opportunities | Count every attempt and repeated hash call, G2 | Diversity differences affect real success |
| Invalid candidates | Retry/filtering hides decoder failures | Invalids consume attempts; invalid reasons reported, G2 | Validity can dominate measured pipeline differences |
| Unequal candidate budgets | Hidden search or repair gives one method extra chances | Complete ledgers, 100 fixed opportunities, G2 | Equal attempts do not equal work |
| Unequal compute budgets | Neural costs can swamp candidate gain | CA0 telemetry and separate Stage II | Stage I cannot infer compute advantage |
| Seed instability | Selected seeds exaggerate reproducibility | Frozen seeds 0/1/2, all outcomes, G4 | Three seeds give limited uncertainty about training variability |
| Decoder failure | A lossy codec or malformed decoding defeats generation | Exact clean corpus plus rejection fixtures, G1-A | Clean correctness does not guarantee good generated validity |
| Representation validity differences | One decoder rejects much more often | Report ValidDecodeRate, diagnostics and matched attempt counts | Attribution to learned hash use requires controls |
| Gaussian/Discrete confounding | Many design components change together | Label end-to-end comparison, report budgets/configurations | No isolated representation causal claim |
| CGGE glyph ambiguity | Distinct characters render identically or classify ambiguously | Frozen glyphs/checksums, ambiguity tests and rules, G1-A | Noisy generated glyphs may remain ambiguous |
| 128-bit extrapolation | Small-size slope misstates full-hash feasibility | Label projections; require separate full-digest evidence | Extrapolation cannot establish asymptotic complexity |
| Training amortization | Large assumed target counts hide upfront work | Report totals, per-target costs, valid reuse population, CA0/Stage II | Deployment volume may differ from assumptions |
| Attack-definition mismatch | Restricted source-derived targets are compared with general attacks | Stage-IV matched target/domain/work review | A comparable SOTA study may remain infeasible |
| Selection and multiplicity | Eligible subsets/budgets/seeds inflate evidence | Prospective A/B inventories and replication correction, G3 | Marginal intervals are not simultaneous intervals |
| Target dependence and uncertainty | Shared checkpoints, constrained groups, finite populations violate broad iid interpretations | Explicit conditional estimand, paired units, independent target RNG, frozen assumption review | Population-level generalization remains limited |
| Shuffled-control mismatch | Inference shuffling or unequal selection weakens the control | True target at inference, matched training/selection, donor audit | Accidental matches and permutation memorization require reporting |

## 13. Reproducibility Requirements

Every primary output must resolve to configuration, seed, checkpoint, dataset manifest, and code revision. Record at minimum:

- Software and library versions, runtime, dependency-lock contents/hash, operating system, hardware, accelerator, numerical precision, determinism settings, and nondeterministic operations.
- Source-generation configuration, PRNG/version, all seed namespaces, draw/duplicate/discard counts, split algorithm/configuration, source/digest counts, ordered target list, dataset hashes and manifests, ownership audit, and representative-selection rule.
- Model/codec configuration, CGGE font/rendering assets and checksums, normalization, condition encoder, training/sampler settings, parameter counts, checkpoint identifiers/checksums, selection history, and model/optimizer/RNG states needed for resumability.
- Git commit plus any uncommitted patch identity, experiment configuration/hash, exact command invocation, start/end timestamps, run ID, protocol/freeze version and date, gate evidence references, and deviations.
- Raw candidate/result ledgers sufficient to reverify every attempt, aligned target outcomes at each prefix, decoder reasons, compute telemetry, statistical configuration, literal hypothesis inventory, paired counts, bootstrap seed/method, raw and Holm-adjusted results, and failure logs.

Binary messages must use lossless serialization such as hexadecimal plus byte length. Model-facing input files must not include evaluator-only metadata. Preserve data lineage and checksums without prescribing a new experiment-management framework. Freeze artifact paths, retention, storage limits, schema versions, and linkage validation. Repository conventions may keep bulky raw data and checkpoints outside version control, provided the retained location and checksums are durable and traceable.

Reproduction includes regenerating the same dataset/split, reconstructing the allowed condition, replaying candidate verification, and recomputing paired statistics from immutable results. Bitwise GPU training reproducibility is not assumed; record any limits and distinguish deterministic verification from stochastic retraining. A source file or an old report does not prove current-protocol compliance.

## 14. Specification Freeze Procedure

The authoritative itemized checklist is §7.29. Freeze all execution-critical TBD/BLOCKING design items into a versioned configuration and decision record before affected scientific execution; record author/reviewer, timestamp, hashes, code revision, and rationale. Do not mark a choice FIXED just because a runtime default exists. Freeze statistical thresholds and inventories before any primary test, including 8-bit tests.

The freeze record must identify which blocking items are design decisions and which require future validation evidence. Design freeze permits implementation/validation of the declared protocol; it does not confer G0, G1, G2, G3, G4, or CA0. Only artifact-backed checks can pass execution gates. Changes before test exposure require versioned amendments; changes after exposure additionally require a defensible uncontaminated evaluation scope. Retain superseded records for audit.

No execution-critical value is imported from older plans by reference. If an existing implementation value is selected, write it explicitly in the freeze record, justify it using training/validation or independent engineering evidence, and check all affected invariants.

## 15. Open Decisions / TBDs

This register groups the unresolved checklist entries; it does not authorize defaults.

**Scientific choices:** dataset construction and count semantics; raw-duplicate policy; group allocation and nested-size design; target sampling/representative rules; CGGE representation/font/rendering/decoder design; BGV decoding thresholds and Gaussian normalization; Gaussian/Discrete architecture, objective, schedule, sampler and training/search budgets; shuffled permutation/collision and selection policy; learned positive-control tasks and thresholds; correctness corpus; validation/checkpoint/stopping rules; representation-ablation budget matching; secondary-diagnostic conventions; inferential population/dependence assumptions, interval level/method, literal Holm family inventories and directions, significance/G3 requirements, adaptive seed-replication multiplicity, complete P1-to-P2 rule and stronger G4 option; overall exit-state scope and aggregation. These must be decided before dependent implementation is treated as canonical or experiments are executed.

**Engineering choices:** seed values/namespaces and PRNG implementations other than fixed model seeds; condition-encoder adapters and model-facing/evaluator schema boundary; continuous-time and reverse-grid implementation details consistent with the scientific decisions; optimizer/batch/precision configurations; hardware/accelerator; software/dependency pinning; timing, NFE, memory and FLOPs instrumentation; mandatory-telemetry availability policy; wall-clock/storage/retention limits; crash/resume and incomplete-run procedure; artifact paths/schemas, checkpoint IDs/checksums, command and code-revision capture; baseline reuse policy and deterministic stream assignment. Some choices, such as precision or batch size, can affect science and must be frozen with the scientific configuration rather than treated as cosmetic.

**Blocking leakage questions:** whether every BGV header/validity/padding component is corrupted and generated without a true-length input; how CGGE's still-unfixed shape/length/padding/decoder preserves the boundary; whether discrete EOS/PAD corruption, initialization, attention masks and reverse sampling are fully target-independent; whether conditioning/collation, shuffle handling, identifiers, RNG assignment, caches, decoders and diagnostic paths expose hidden source information; and whether cross-size reuse exposes evaluation conditions or source structure. The required safe behavior is fixed, but implementation compliance has not been established. Close these questions with the traces and mutation checks in §7.12, not assertions.

## 16. Claim Language Guide

The following are templates for future reporting, not observations made in this document.

**Acceptable after the corresponding G3/G4 evidence:** “Under the tested source distribution, truncated digest size, candidate budget, architecture, and training budget, the hash-conditioned model showed a reproducible increase in preimage success relative to the preregistered controls.” Include the setting, effects, confidence intervals, corrected tests and seed outcomes.

**Acceptable for sign-only P2:** “The paired effects against both controls were positive for seeds 0, 1 and 2 in this setting; the reported uncertainty and corrected tests do not establish a stronger claim.” Do not use this sentence if replication was incomplete or a sign was not positive.

**Unacceptable from Stage I:** “Diffusion models invert MD5.” “Diffusion models break MD5.” “Diffusion models reduce MD5 preimage complexity.” “Diffusion models outperform state-of-the-art MD5 preimage cryptanalysis.” Also unacceptable are arbitrary-target, SHA-256, full-digest or compute-advantage claims inferred solely from candidate budgets.

**Acceptable negative result:** “Under the tested source distribution, digest size, candidate budget, architecture, and training budget, no reproducible hash-conditioned advantage was observed.” State whether this means an interpretable null/unstable result or a blocked pipeline. A Stage-I negative result is not proof that diffusion-based inversion is impossible.

**Acceptable gate failure:** “The pipeline did not satisfy its preregistered positive-control requirement, so its hash experiment was blocked; the scientific hypothesis was not tested by that pipeline.”

## Appendix A. Repository Inspection and Implementation Discrepancies

Inspection used code revision `a9bd58c2a0e7bf3fb9ee47e1dfa0a015c118113c` with an initially clean working tree. The repository contains `src/diffusion_hash_inv/`, `tests/`, `examples/`, `pyproject.toml`, `uv.lock`, planning files, and a local experiment archive. This was a read-only compatibility inspection; existing tests and scientific jobs were not executed for this documentation task. Neither old text nor archived artifacts have been accepted as gate evidence here.

| Observed implementation/context | Protocol consequence and future required work |
|---|---|
| Pre-existing planning files contain other matrices, larger data sizes, Direct Bits, SHA-256 and known-length directions | Preserve this five-pipeline, MD5-only primary scope; reconcile entry points/configuration labels before execution. Other files are not canonical authority. |
| `dataset.py` has unique-message rejection sampling, a greedy whole-group split defaulting to 0.8/0.1/0.1, and a separate exact-quota constructor | Neither constructor is automatically selected. The ratio does not express the stipulated 10,000/1,000/1,000 allocation. Freeze count semantics, group construction and nested-size handling, then adapt the selected implementation. |
| `dataset.py` stores full digests, messages and length in records | Retain only on the evaluator/data side; introduce or audit a narrow prefix-only generation boundary and mutation checks. |
| `conditioning.py` supports both algorithms and an optional length feature; `runner.py` supports legacy captions and several condition modes | Restrict primary runs to exactly the allowed MD5 bits with no length feature; explicitly freeze the equivalent condition encoder. Merely leaving a default off is not a leakage audit. |
| `runner.py` calls `_conditions` during both training and generation, and that helper applies its shuffle mode to the supplied records | Separate shuffled training/selection behavior from evaluation. Every trained model must receive the actual target during primary inference; inference-time donor permutation cannot implement the required control. |
| `encoding/bgv.py` already implements the required geometry and MSB-first row-major mapping | Reuse the clean mapping; freeze threshold/normalization/strict-decoding behavior and validate target-independent generation under the selected model. |
| `encoding/cgge.py` contains a fixed embedded glyph table, dimensions, classifier and defaults | Treat these as candidate engineering choices. Do not import font/dimensions/thresholds as scientifically frozen; audit ambiguity, deterministic rendering and length handling after selection. |
| `encoding/tokens.py` uses separate payload/EOS/PAD/MASK IDs and strict grammar; `discrete.py` has continuous-time training draws and a finite reverse grid | Token conventions are reusable. Current continuous training is not itself a finite-training-schedule conflict, but architecture, loss, reverse schedule and all entry points require freeze and validation. |
| `models.py` and `runner.py` supply Gaussian architecture, schedule, optimizer, training and sampling defaults | Defaults do not select the scientific model. Freeze them explicitly or change implementation; verify forward/reverse consistency and full-image initialization. |
| `study_statistics.py` requires all three seeds, performs per-seed family correction, and uses fixed 0.05/95% decisions; `evaluation.py` supplies paired statistics | Reconcile with staged seed-0 eligibility, prospective families, and still-unfixed G3/interval criteria. Existing numerical thresholds are not inherited. Audit exact-test and bootstrap edge cases after freeze. |
| `positive_control.py` and `preflight.py` contain engineering/positive-control routines and their own labels and defaults | Map to this document's G1-A/G1-B, ensure each applicable actual learned pipeline is covered, and register tasks/thresholds. Presence of routines does not show any gate passed. |
| `protocol_gates.py` and candidate/evaluation modules provide reusable audit/ledger components | Extend or reconcile schema/criteria as required; prove exact 100-attempt streams, control alignment, leakage limits and traceability before scientific use. |

All required implementation changes are future work. This document does not certify the current experiment command as compliant.

## Appendix B. Document Consistency and Numerical Audit

The audit below concerns the written specification, not executed scientific validation. An unresolved choice is retained explicitly in the freeze checklist; no audit PASS converts it into a frozen value or a passed scientific gate.

| Audit category | Document result | Checked invariant |
|---|---|---|
| Pipeline matrix | PASS | Exactly five primary IDs, three primary sizes, 15 settings; CGGE Printable-only |
| Representation definitions | PASS | BGV $[2,32,128]$; CGGE unresolved details explicit; vocabulary 97/259; PAD distinct from NUL |
| Leakage constraints | PASS | Hidden length/EOS/PAD/suffix forbidden; target-independent generation required; unresolved implementation evidence blocking |
| Dataset/split design | PASS | Both separations, nested-size choice and count-semantics conflict explicit; no unfrozen algorithm presented as executable |
| Candidate accounting | PASS | Shared 100-attempt stream; invalids/duplicates count; no success-conditioned retries |
| Controls | PASS | Every learned pipeline requires source-prior random, shuffled training and actual conditional positive control |
| Statistics | PASS | Paired unit, one-sided exact McNemar, 10,000 paired resamples, separate Holm A/B; unresolved thresholds/selection explicit |
| Claim boundaries | PASS | Sign-only versus statistical evidence distinguished; P2 does not imply computational/full-MD5/SOTA evidence |
| Stage separation | PASS | Later stages require separate protocols; fresh future SOTA review |
| Numerical sanity | PASS | Arithmetic and idealized success expression checked; no measured performance asserted |
| Math delimiters | PASS | Dollar-delimited inline/display math; balanced delimiters and standard LaTeX |

Independently checked relationships are

$$
\begin{aligned}
94+3&=97,&256+3&=259,&31+1&=32,\\
5\times3&=15,&2^8&=256,&31-4+1&=28,\\
2\times4&=8,&4\times4&=16,&4\times8&=32,\\
8\times16&=128,&15\times3\times2&=90,&5\times2\times1\times2&=20.
\end{aligned}
$$

For BGV, $2\times4=8$ gives cell height and $4\times4=16$ gives cell width from logical glyph dimensions and pixel-block size. Four cell rows give image height 32; eight cell columns give image width 128; the four-by-eight slot grid contains 32 slots.

For independent trials with per-attempt success probability $p$, failure throughout $K$ trials has probability $(1-p)^K$; its complement is $1-(1-p)^K$. Substituting the idealized $p=2^{-q}$ gives the stated formula. This is an analytic sanity check, not an experimental result. The optional illustrative family counts are $15\times3\times2=90$ and $5\times2\times1\times2=20$; neither is an adopted family inventory.

**Math delimiter audit: PASS**

**Readiness: RESEARCH PLAN DOCUMENTED — SPECIFICATION FREEZE STILL REQUIRED**
