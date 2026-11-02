# NEUROGEN v0.3 — Verification Report

**Date:** 2026-11-01
**Release:** neurogen-v0.3.zip
**Baseline:** NEUROGEN v0.2.1 (frozen, untouched)

## Provenance

| Artifact | SHA-256 | Size |
|---|---|---|
| Baseline `neurogen-v0.2.1.zip` (original, frozen) | `61a871a46a76b5d6d3c01a79aebad07ecb8183e004d1388ef7210b92abc60357` | 28,919 bytes |
| Release `neurogen-v0.3.zip` | recorded in the external release record (`reports/verification/`) — a document inside the archive cannot contain its own archive hash | 78,166 bytes |

The baseline archive was never modified. v0.3 was developed in a separate
working tree copied from the extracted baseline. All v0.2.1 modules are
reused; only `parser.py`, `ast_nodes.py`, `compiler.py`, `interface.py`
received minimal extensions, and `__init__.py` was updated. New modules:
`src/structure.py`, `src/pipeline.py`, `src/pipeline_engine.py`,
`src/errors.py`.

## Environment

- OS: Linux x86_64 (Linux-7.0.0-39-generic)
- Python: 3.12.3
- numpy: 1.26.4 (required dependency)
- CPU: 4 cores

## Reproduction commands

All commands run from the extracted `neurogen-v0.3/` directory:

```bash
PYTHONPATH=src python3 tests/test_neurogen.py            # v0.2.1 regression
PYTHONPATH=src python3 tests/test_pipeline_functional.py # functional
PYTHONPATH=src python3 tests/test_pipeline_negative.py   # negative
PYTHONPATH=src python3 tests/test_mutation_compat.py     # mutation compat
PYTHONPATH=src python3 tests/test_pipeline_stress.py     # stress
PYTHONPATH=src python3 demos_v03.py                      # six demos
PYTHONPATH=src python3 demo.py                           # v0.2.1 demos
PYTHONPATH=src python3 benchmark_v03.py                  # benchmarks
```

## Results

| Suite | Executed | Passed | Failed |
|---|---|---|---|
| v0.2.1 regression (`test_neurogen.py`, 20 groups) | 20 | 20 | 0 |
| Functional (`test_pipeline_functional.py`) | 23 | 23 | 0 |
| Negative (`test_pipeline_negative.py`) | 28 | 28 | 0 |
| Mutation compatibility (`test_mutation_compat.py`) | 5 | 5 | 0 |
| Stress: valid pipeline executions (seed 20261101) | 5,000 | 5,000 | 0 |
| Stress: invalid-connection rejections (seed 20261102) | 5,000 | 5,000 | 0 |
| Stress: multi-structure graphs (seed 20261103) | 1,000 | 1,000 | 0 |
| Stress: repeated changing inputs (seed 20261104) | 1,000 | 1,000 | 0 |
| Stress: cache invalidation (seed 20261105) | 1,000 | 1,000 | 0 |
| Demonstrations (`demos_v03.py`) | 6 | 6 | 0 |
| v0.2.1 demos (`demo.py`) | 5 | 5 | 0 |

Total: **13,087 checks executed, 13,087 passed, 0 failed.**

## What was verified

- Independent structure compilation and execution, via the new structure
  interface and via the legacy genome-level `OUTPUT` path (identical results).
- Real computed values transported between structures (no symbolic links,
  no hardcoded outputs; demo expectations computed independently, e.g. with
  numpy for the vector pipeline).
- Multi-input, multi-output, branching, and shared-dependency pipelines.
- Copy-on-fan-out: numpy arrays provably share no writable memory between
  branches (mutation test); scalars shared safely.
- Cache correctness: hits on identical inputs, misses on changed inputs,
  disabled mode, and `CONNECT` structures never served from cache (memory
  writes execute on every run — verified via per-run memory dumps).
- Failure isolation: diamond pipeline with one failing branch — the other
  branch completes, the join is `blocked`, the pipeline reports `failed`.
- Compile-time rejection of: unknown structures/inputs/outputs, type
  mismatches (only INT→FLOAT promotion allowed), duplicate connections,
  circular dependencies (2-cycle, 3-cycle, self-loop), unreachable
  structures, oversized pipelines, malformed syntax.
- Runtime defense in depth: untyped (ANY) values re-validated on injection.
- Structured traces: order, status, summarized inputs/outputs, connections
  traversed, cache states, errors with stage identity, per-stage durations.
- The v0.2.1 `ControlledEvolver` still produces valid, executable graphs;
  mutated structures re-interface and run in pipelines; incompatible
  mutated interfaces are rejected at connection validation.

## Bugs found and fixed during development

1. **CONNECT pruning:** `finalize_graph` pruned `CONNECT` (memory-write)
   nodes when they were not ancestors of a declared output, silently
   dropping the write. Fixed: CONNECT nodes are retained as side-effect
   roots in `compile_structure_body`.
2. **Missing output node:** independently compiled graphs had no `output`
   node, so `Runtime.execute` refused to run. Fixed: declared outputs are
   wired through `output` nodes in `compile_structure_body`.
3. **VECTOR/MATRIX uncallable:** the v0.2.1 parser tokenizes `VECTOR` as a
   keyword, making the registered `VECTOR`/`MATRIX` operators unusable in
   `LET` position. Fixed minimally: those two keywords are accepted as
   operator names; all other keywords keep v0.2.1 behavior.
4. **Trace plumbing:** engine initially stashed per-structure memory/duration
   in attributes; refactored to return them directly from `_run_structure`.

## Known issues / incomplete work

- None blocking. See `docs/LIMITATIONS_v0.3.md` for explicit limitations
  carried over from v0.2.1 and new v0.3 boundaries.

## Backward compatibility

All 20 v0.2.1 verification groups pass unmodified against the v0.3 tree.
All 7 v0.2.1 example programs still parse; all 5 v0.2.1 demos execute with
identical results. The legacy `Neurogen` API is untouched; `NeurogenPipeline`
is additive.
