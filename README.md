# NEUROGEN v0.3

**A language-native cognitive architecture: concept graphs, typed pipelines, and reproducible reasoning experiments in Python.**

Blueprint artwork: published with the website edition on the [Dimon Global Group — Systems page](https://mammon-h.github.io/).

NEUROGEN is an AI-native programming language and computational environment.
Instead of bolting cognition onto a general-purpose language, it treats
computational architectures, memory, reusable motifs, and controlled program
development as first-class language concepts. You declare `STRUCTURE`s that
compute, and `PIPELINE`s that connect them — the language validates the
wiring at compile time and exchanges real, typed values at runtime.

A Dimon Global Group release.

## What v0.3 adds: cross-structure typed data exchange

Independent `STRUCTURE`s declare typed inputs and named outputs and exchange
real computed values through validated `PIPELINE` declarations:

```neurogen
STRUCTURE perception {
    INPUT image : TENSOR
    LET features = NORMALIZE(image)
    OUTPUT features
}

STRUCTURE reasoning {
    INPUT evidence : TENSOR
    LET score = REDUCE(evidence)
    OUTPUT score
}

PIPELINE main {
    CONNECT perception.features TO reasoning.evidence
    OUTPUT reasoning.score
}
```

- **Compile-time connection validation** — exact types; only `INT → FLOAT`
  promotion allowed. Cycles, duplicates, unknown endpoints, and unreachable
  structures are rejected with diagnostics before anything runs.
- **Runtime data exchange** — copy-on-fan-out branching, shared-dependency
  caching (memory-writing structures are never cached), failure isolation
  (`failed` / `blocked` per-structure status), and structured execution
  traces.
- **Fully backward compatible** — the entire v0.2.1 language, runtime,
  operators, and API are preserved; every original test passes unmodified.

## Quick start

Requires **Python 3.8+** and **numpy** (`pip install numpy`).

```bash
PYTHONPATH=src python3 demos_v03.py

PYTHONPATH=src python3 -c "
from interface import NeurogenPipeline
ngp = NeurogenPipeline().load(open('examples/pipelines/demo1_basic.ng').read())
print(ngp.run('main', {'stage_a.x': 5.0}).outputs)
"
```

## Verification

The release ships with its full test suite — run it yourself:

```bash
PYTHONPATH=src python3 tests/test_neurogen.py            # v0.2.1 regression (20 groups)
PYTHONPATH=src python3 tests/test_pipeline_functional.py # 23 functional tests
PYTHONPATH=src python3 tests/test_pipeline_negative.py   # 28 negative tests
PYTHONPATH=src python3 tests/test_mutation_compat.py     # 5 mutation tests
PYTHONPATH=src python3 tests/test_pipeline_stress.py     # 13,000 stress scenarios
```

## Benchmarks (measured, reproducible)

From `benchmark_v03.py` on the verification environment (Linux, Python
3.12.3, numpy 1.26.4, 4 CPUs — do not compare across environments; see
`docs/BENCHMARKS_v0.3.md` for full tables):

- Compilation scales approximately linearly: 0.25 ms for a 1-structure
  chain, 3.23 ms for a 20-structure chain (mean).
- Execution overhead is roughly 35–45 µs per structure; a 20-structure
  chain executes in 0.76 ms mean.
- A warmed cache cuts 5-structure chain latency by ~39% (0.211 ms →
  0.129 ms) at a 100% hit rate.
- Peak memory growth across the full benchmark session: +256 KB.

## Documentation

- `docs/SPEC.md` — the v0.3 language additions
- `docs/ARCHITECTURE_v0.3.md` — how the pipeline engine is built
- `docs/VERIFICATION_v0.3.md` — what was verified, and how
- `docs/BENCHMARKS_v0.3.md` — full benchmark tables
- `docs/LIMITATIONS_v0.3.md` — known limitations, stated plainly
- `docs/ARCHITECTURE_AUDIT_v0.2.1.md` — the independent audit of the
  previous generation

## Honest limitations

v0.3 does not pretend to be more than it is. Memory is in-process and does
not survive restarts; the learning module is a numerical helper, not
autonomous learning; there is no neural-network integration and no
OS-level sandboxing. The full list is in `docs/LIMITATIONS_v0.3.md` —
read it before building on this.

## Version note

v0.3 is a complete, frozen public generation of NEUROGEN, released as-is
for study, use, and extension. NEUROGEN development continues privately:
a successor generation exists that surpasses v0.3 and is not part of this
repository. The v0.2.1 history is preserved in the docs and in the
regression suite, which still passes in full.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Dimon Global Group.
