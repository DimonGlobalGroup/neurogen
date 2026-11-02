# NEUROGEN v0.3 — Proposed Architecture: Cross-Structure Execution

**Status:** APPROVED and IMPLEMENTED (2026-11-01). This document was the reviewed
design for the v0.3 build; the implementation in `src/` follows it, with the
five approved refinements (selective copy-on-fan-out, CONNECT cache bypass,
failure isolation, independent structure execution, structured tracing).
**Date:** 2026-11-01
**Author:** Muse (implementation engineer)

This document proposes the v0.3 design. It adapts the conceptual architecture
from the handoff to the *actual* v0.2.1 implementation, which differs in one
central respect: **v0.2.1 flattens all structures into a single shared graph.**
There is no independent structure compilation today. The design below makes the
minimal changes needed to turn structures into genuinely independent,
connectable computational units.

## 1. Core design decisions

### 1.1 Independent structure compilation (the central change)

The v0.2.1 compiler flattens every `STRUCTURE` into one shared graph with one
genome-level `OUTPUT`. v0.3 will compile each `STRUCTURE` into its **own**
`ComputationalGraph` (reusing the existing compiler and graph machinery), with
a declared interface:

- **Inputs:** the existing `INPUT name : TYPE` statements (unchanged syntax).
- **Outputs:** new explicit `OUTPUT name` statements *inside* structure bodies,
  referencing names bound by `LET`/`CONST`/`INPUT`. Minimal syntax — reuses the
  existing OUTPUT keyword at a new scope.

Backward compatibility: a genome with a single structure and genome-level
`OUTPUT name` continues to work exactly as today (compiled standalone,
executed by the existing `Runtime`).

### 1.2 PIPELINE declaration

New top-level statement (parser extension, minimal):

```neurogen
PIPELINE main {
    CONNECT perception.features TO reasoning.evidence
    CONNECT reasoning.decision TO evaluation.score
    OUTPUT evaluation.result
}
```

- `CONNECT <structure>.<output> TO <structure>.<input>` — dotted references,
  resolved against declared interfaces at compile time.
- `OUTPUT <structure>.<output>` inside a pipeline selects final outputs.
- Structures may appear in a pipeline without knowing about it.

### 1.3 Pipeline compiler and intermediate representation

New module `src/pipeline.py`:

- `PipelineIR`: pipeline id, structure references (compiled graphs + interfaces),
  connection records, dependency edges, topological execution order, output
  declarations, validation metadata.
- Validation at compile time: source/destination structures exist; named
  outputs/inputs exist; types compatible (see §1.4); no duplicate conflicting
  connections; dependency graph acyclic (cycles rejected with diagnostics);
  unreachable structures reported.
- Reuses compiled structure graphs — no duplication of graph data.

### 1.4 Typed connections

Connection validation rules:

- Exact dtype match required by default.
- `INT → FLOAT` promotion allowed (consistent with v0.2.1's existing input
  rule).
- All other mismatches (e.g. VECTOR → FLOAT, FLOAT → BOOL) rejected **at
  compile time** with a clear diagnostic naming source, destination, and types.
- Runtime still re-validates values before injection (defense in depth);
  a value that fails at runtime raises `StructureInputError` and blocks
  downstream stages.

### 1.5 Pipeline engine (runtime data exchange)

New module `src/pipeline_engine.py`:

- Executes structures in topological dependency order using the existing
  `Runtime` per structure.
- A structure's declared outputs are read as `NeuroValue`s from its graph's
  named bound nodes, validated, and injected as the downstream structure's
  inputs (which become entries in its runtime context).
- Multiple upstream structures feeding one downstream structure: the downstream
  structure executes only after all required inputs are available.
- Branching (one output → many inputs): **copy-on-fan-out.** Because
  INDEX/SLICE can produce numpy views, downstream structures receive deep
  copies, preventing accidental cross-stage mutation. Documented in the
  language spec.
- Per-structure status: `success | failed | blocked | skipped`. Pipeline-level
  status. Upstream failure blocks downstream stages explicitly; the pipeline
  never reports success when a required stage failed. No zero-value
  substitution.
- Execution limits: max structures per pipeline, max connections, max
  dependency depth, per-structure timeout (reusing existing `Runtime` limits),
  total pipeline budget. Tensor-element caps remain a documented limitation.

### 1.6 Caching

- Cache key: (structure name, structure definition hash, input fingerprint,
  execution settings).
- Inputs change → fingerprint changes → stale results never reused.
- Cache hit/miss recorded in the trace; behavior testable with caching on/off.
- Structures with side effects (`CONNECT` memory writes): cache key includes a
  marker so replays are consistent; documented.

### 1.7 Execution tracing

`PipelineTrace`: pipeline id, execution id, per-structure execution order,
status, inputs received (summarized, not full tensor dumps), outputs produced
(summarized), connections traversed, cache hits/misses, errors with stage
identity, per-stage durations. Optional controlled inspection of full values.

### 1.8 Error model

New exception hierarchy (all subclassing a common `NeurogenError` where
sensible, without disturbing existing `CompileError`/`RuntimeError`/`ParseError`
/`OperatorError`):

- `StructureError` (compilation: unknown structure, bad interface)
- `ConnectionError` (unknown input/output, type mismatch, duplicate)
- `PipelineError` (cycle, unreachable structure, malformed pipeline)
- `StructureInputError` / `StructureExecutionError` (runtime)

## 2. Module plan (working tree)

Copy of all v0.2.1 sources, byte-identical except where noted, plus:

| Module | New / Changed | Responsibility |
|---|---|---|
| `src/parser.py` | extended | `PIPELINE` block, `CONNECT a.b TO c.d`, pipeline-scope `OUTPUT`; structure-scope `OUTPUT name`. |
| `src/ast_nodes.py` | extended | `Pipeline`, `PipelineConnect`, `StructureOutput` nodes; `PipelineIR` dataclass. |
| `src/structure.py` | **new** | Structure interface (inputs/outputs/types), independent compilation of one structure → its own `ComputationalGraph`, output extraction. |
| `src/pipeline.py` | **new** | Pipeline compiler: reference resolution, connection validation, dependency graph, topological scheduling, IR construction. |
| `src/pipeline_engine.py` | **new** | Pipeline runtime: ordered execution, value transport, caching, tracing, error propagation, budgets. |
| `src/compiler.py` | extended minimally | Expose a structure-scoped compile entry point used by `structure.py`; existing `compile_program` behavior unchanged. |
| `src/interface.py` | extended minimally | `NeurogenPipeline` facade alongside existing `Neurogen`; existing API unchanged. |
| `src/runtime.py`, `operators.py`, `neuro_types.py`, `memory.py`, `evolution.py`, `learning.py` | unchanged | Reused as-is. Evolution continues to mutate single-structure graphs; v0.3 tests will verify mutation compatibility with pipeline interfaces. |

No v0.2.1 module is rewritten. The baseline archive remains untouched.

## 3. Why this design (alternatives considered)

- **Alternative: keep flat compilation and simulate pipelines in Python.**
  Rejected — it would be connecting Python functions, not extending the
  NEUROGEN language. The handoff's definition of success explicitly forbids this.
- **Alternative: inline all structures into one mega-graph.**
  Rejected — destroys independent executability and inspectability; structures
  could not be tested alone or reused across pipelines.
- **Alternative: message-passing / ports runtime separate from the graph.**
  Rejected as over-engineering — the existing `Runtime` already executes rooted
  graphs correctly; the pipeline engine is a thin, inspectable orchestration
  layer on top of proven machinery.

## 4. Open questions for review

1. Copy-on-fan-out vs. shared references for branching — proposed: copy
   (safety), with the option to revisit if benchmarks show meaningful cost.
2. `OUTPUT` inside structure bodies reuses the keyword at a new scope — any
   objection to this vs. a new keyword (e.g. `EXPORT`)?
3. Type promotion limited to INT→FLOAT — keep it that tight for v0.3?
