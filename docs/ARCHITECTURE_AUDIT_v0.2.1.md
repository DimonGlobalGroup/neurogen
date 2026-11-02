# NEUROGEN v0.2.1 — Architecture Audit

**Date:** 2026-11-01
**Auditor:** Muse (implementation engineer)
**Baseline archive:** `baseline/neurogen-v0.2.1.zip`
**SHA-256:** `61a871a46a76b5d6d3c01a79aebad07ecb8183e004d1388ef7210b92abc60357`
**Archive size:** 28,919 bytes
**Extracted size:** 136K, 18 files, 2,104 total lines

Environment: Python 3.x, numpy 1.26.4, Linux x86_64.

The archive is a genuine Zip archive (compression method=store) containing a single
top-level directory `neurogen/`. The original archive has not been modified.
All inspection was performed on the extracted working copy.

## Verification results (executed, not copied)

Ran `PYTHONPATH=src python3 tests/test_neurogen.py` on the extracted baseline:

- 20/20 verification groups passed.
- Operator stress: 5,000 randomized executions, 0 failures.
- End-to-end batch: 50 runs, all returned numeric results.
- Numerical extremes: 1e10 / -1e10 / 1e-12 / -1e-12 / 0.0 / ±1.0 passed.
- All 32 registered operators smoke-tested.
- `demo.py`: 5 demos executed (math_ops → 1.0; vector_ops → 2.0925; logic → True;
  pipeline → memory-write wrapper; motif_reuse → 7.3106).
- `benchmark.py` (100 runs each): math_ops avg 0.065 ms; vector_ops avg 0.083 ms;
  pipeline avg 0.107 ms.

Discrepancies vs the supplied verification report:
- The report cites microbenchmarks of approximately 0.03–0.08 ms per sample
  program. math_ops (0.065) and vector_ops (0.083) are within/at range; the
  pipeline example measured 0.107 ms, slightly above the reported range. This is
  a minor deviation, not a failure; single-machine timing noise cannot be ruled out.
- All other recorded categories reproduce exactly.

## Source architecture (actual, from inspection)

### Pipeline (as implemented)

```
SOURCE → tokenizer (regex, per line) → Parser → AST (dataclasses)
  → compile_program → ComputationalGraph (nodes + edges, rooted at output)
  → Runtime.execute (Kahn topological order, per-node eval, trace)
  → computed result
```

### Modules

| File | Lines | Responsibility |
|---|---|---|
| `src/parser.py` | 257 | Line tokenizer + recursive-descent parser. Keywords uppercase by convention. Statements: GENOME, MEMORY, STRUCTURE, MOTIF, INPUT (optionally `: TYPE`), CONST (number/string literals), LET (`= OPERATOR(args)` / `= identifier` / `= literal`), bare operator calls (CallStmt), USE, CONNECT, TRANSFORM, EVALUATE, OBSERVE, ADAPT, OUTPUT (genome-level, single structure). |
| `src/ast_nodes.py` | 146 | Dataclass AST nodes. `ComputationalGraph`: nodes dict, edges list, output_id, memory_kind, motifs, structure_nodes, structure_outputs; `topological_order()` via Kahn's algorithm, raises ValueError on cycle. |
| `src/compiler.py` | 303 | `compile_program`: single flat compilation. All structures compile into ONE shared graph namespace (`name_to_node` is per-structure local map, but nodes/edges are global). INPUT → `input` node; CONST → `const` node; LET/Call → operator node with `("ref", node_id)` / `("literal", value)` args; USE inlines motif bodies with first-INPUT consumption. OUTPUT wires a single structure's root to an `output` node. Prunes unreachable nodes, then cycle-checks. |
| `src/runtime.py` | 174 | `Runtime.execute`: Kahn order, `max_steps` node limit, `timeout_s` checked between nodes. `input` nodes: missing input raises RuntimeError; declared type enforced with INT→FLOAT promotion allowed. `CONNECT` writes payload to whichever memory backend is configured. Operator nodes: resolves ref/literal args; if resolved args don't match operator arity, falls back to predecessor values (documented heuristic — see §Observations). OperatorError wrapped into RuntimeError. Full per-node trace recorded. |
| `src/operators.py` | 173 | 32 operators in REGISTRY (all uppercase). Math (9): ADD SUBTRACT MULTIPLY DIVIDE POWER MODULO MINIMUM MAXIMUM. Logic (7): AND OR NOT XOR EQUAL GREATER LESS. Transforms (6): TANH SIGMOID RELU SOFTMAX NORMALIZE CLAMP. Data (10): VECTOR MATRIX INDEX SLICE CONCATENATE REDUCE MAP TRANSFORM AGGREGATE MERGE ROUTE. numpy-backed. DIVIDE raises OperatorError on zero denominator. `MAP` uses a generator-throw idiom for unknown targets. |
| `src/neuro_types.py` | 117 | `DType` enum: BOOL INT FLOAT STRING VECTOR MATRIX TENSOR OBJECT GRAPH_REF MEMORY_REF ANY. `NeuroValue` dataclass (data, dtype, shape, meta); `from_python` infers dtype (lists/tuples → numpy arrays, ndim decides VECTOR/MATRIX/TENSOR). `promote_type`, NUMERIC/SCALAR sets. |
| `src/memory.py` | 93 | `AssociativeMemory` (tag-scored retrieval, strength, access counts) and `KeyValueMemory` (plain dict). Both in-process only. |
| `src/evolution.py` | 144 | `ControlledEvolver`: deep-copies graph; mutations: `_add_transform` (insert RELU), `_duplicate_node` (parallel copy), `_prune_redundant` (bypass single-parent/single-child `transform` nodes). `_rewire` is defined but never selected by `mutate`. Always re-validates (reachability prune + topological check). Mutation quality is NOT measured — structural validity only. |
| `src/learning.py` | 43 | `numerical_gradient` (central differences) and `simple_optimize` (gradient descent on a Python objective). Deliberately modest. Not autonomous learning. |
| `src/interface.py` | 65 | `Neurogen` facade: load → parse → compile → Runtime; run(inputs); evolve(intensity); inspect(); source_of_truth(). |

### External dependency

`numpy` (1.26.4 in this environment) is required by `neuro_types.py` and
`operators.py`. v0.3 must record this dependency in install instructions.

## Key structural observations for v0.3 design

1. **Structures are NOT independent in v0.2.1.** Although the language has a
   `STRUCTURE` keyword, the compiler flattens every structure into a single
   shared graph namespace, and a genome has exactly one OUTPUT (one structure
   root). Genuine cross-structure data exchange does not exist; motifs are
   inlined into the same graph. This is the central gap v0.3 must fill.
2. **`OUTPUT` is genome-level only.** There is no mechanism to declare a
   structure's named outputs. v0.3 needs explicit per-structure output
   declarations (minimal syntax extension).
3. **No `PIPELINE` concept.** Connections between structures exist only
   symbolically (CONNECT writes to memory). v0.3 needs a pipeline declaration
   with real runtime value transport.
4. **Implicit data-flow wiring.** When an operator call's args are all literals
   (no refs), the compiler wires `current_values` as predecessors, and the
   runtime falls back to predecessor values on arity mismatch. v0.3 structure
   compilation must keep this behavior for backward compatibility while making
   explicit INPUT/OUTPUT interfaces authoritative.
5. **Value model for fan-out.** `NeuroValue` wraps numpy arrays; INDEX/SLICE
   can produce views. A v0.3 design decision is needed on copy vs. share for
   branching (recommendation in the v0.3 architecture doc: copy on fan-out).
6. **Timeout granularity.** Timeouts are checked between node evaluations, not
   inside an operator. Retained as a documented limitation.
7. **Resource limits.** Node-count limit and execution timeout exist; no
   tensor-element caps, no memory budgets, no OS sandboxing. Retained as
   documented limitations.

## Limitations confirmed as explicit (must remain explicit in v0.3 docs)

- Memory is in-process; not durable across restarts.
- `learning.py` is a numerical-gradient helper, not intelligent learning.
- Mutation quality / behavioral improvement not established.
- No tensor-size or memory budgets.
- No independent-structure data exchange (v0.3 objective).
- No neural-network integration.
- No general runtime self-modification.
- No OS-level isolation (language-level restrictions only).

## Baseline preservation

Original archive untouched at `baseline/neurogen-v0.2.1.zip`. SHA-256 recorded
above. v0.3 development will occur in `working/neurogen-v0.3/` as a separate
tree. The baseline extraction is for inspection only.
