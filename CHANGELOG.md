# Changelog

## v0.3 — 2026-11-01 (public release)

- Cross-structure typed data exchange: `STRUCTURE`s declare typed inputs
  and named outputs; `PIPELINE` declarations connect them.
- Compile-time connection validation (exact types, `INT → FLOAT`
  promotion only); cycles, duplicates, unknown endpoints, and unreachable
  structures rejected with diagnostics.
- Runtime: copy-on-fan-out branching, shared-dependency caching
  (memory-writing structures never cached), failure isolation
  (`failed` / `blocked` per-structure status), structured execution
  traces.
- Test suite: 20 v0.2.1 regression groups, 23 functional, 28 negative,
  5 mutation-compatibility tests, 13,000 stress scenarios.
- Benchmarks published in `docs/BENCHMARKS_v0.3.md`; limitations stated
  in `docs/LIMITATIONS_v0.3.md`.

## v0.2.1 — verification & reliability (history preserved in docs)

- KeyValueMemory receives `CONNECT` writes; typed/missing inputs
  enforced; DIVIDE raises on division by zero; mutation tests execute
  the mutated graph; expanded negative tests. No syntax changes.
