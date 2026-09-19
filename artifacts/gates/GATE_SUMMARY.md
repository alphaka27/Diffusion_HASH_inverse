# Scientific gate status

| Gate | Current scientific status | Actual completed checks |
|---|---|---|
| G0 | BLOCKED_SPECIFICATION | 12 engineering sanity manifests/splits PASS; no scientific dataset frozen |
| G1 | PARTIAL | 9,636 codec round-trips plus 94 single glyph tests PASS; verifier exhaustive PASS; learned reversible controls NOT RUN |
| G2 | NOT_RUN | Exact-budget, ordering, version and negative-case unit tests PASS; sanity self-comparisons PASS |
| G3 | NOT_RUN | One-/two-sided exact McNemar, 10,000-resample paired bootstrap and Holm exercised on synthetic binary fixture only |
| G4 | NOT_RUN | Seed aggregation implementation checked on synthetic fixture; no scientific model seeds trained |

The legacy G0–G6 gate namespace is unrelated and its old PASS labels are not imported. No L0–L4 efficacy evidence is assigned to unexecuted experiments.
