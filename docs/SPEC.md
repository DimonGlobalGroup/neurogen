# NEUROGEN Language Specification v0.2

**Purpose**  
NEUROGEN is an AI-native programming language and computational environment.  
Architectures, memory, relationships, motifs, and controlled development are first-class concepts.

## Status

v0.2.1 focuses on **real execution**:

- Operators perform actual mathematical, logical, and data computations.
- Values carry type information (BOOL, INT, FLOAT, VECTOR, MATRIX, TENSOR, …).
- Computational graphs are dependency-ordered and execute deterministically.
- Structures and motifs exchange real data.
- Regression compatibility with v0.1 structural programs is maintained.

## Core Syntax

```neurogen
GENOME "Name"

MEMORY associative          # or keyvalue

MOTIF feature_gate {
    INPUT x : FLOAT
    LET gated = SIGMOID(x)
}

STRUCTURE pipeline {
    INPUT data : VECTOR
    USE feature_gate
    LET norm = NORMALIZE(data)
    LET act = RELU(norm)
    LET score = REDUCE(act)
    CONST thresh = 0.5
    LET decision = GREATER(score, thresh)
    CONNECT memory
}

OUTPUT pipeline
```

### Statements

| Statement | Meaning |
|-----------|---------|
| `GENOME "name"` | Program identity |
| `MEMORY associative\|keyvalue` | Memory backend |
| `STRUCTURE name { … }` | Named computational structure |
| `MOTIF name { … }` | Reusable computational motif |
| `INPUT name [: TYPE]` | Typed input |
| `CONST name = literal` | Constant |
| `LET name = OPERATOR(args…)` | Bind result of operator |
| `OPERATOR(args…)` | Direct operator call |
| `USE motif` | Inline a motif |
| `CONNECT target` | Store into memory |
| `OUTPUT structure` | Root of the graph |

### Supported Operators (v0.2)

**Math:** ADD, SUBTRACT, MULTIPLY, DIVIDE, POWER, MODULO, MINIMUM, MAXIMUM  
**Logic:** AND, OR, NOT, XOR, EQUAL, GREATER, LESS  
**Transforms:** TANH, SIGMOID, RELU, SOFTMAX, NORMALIZE, CLAMP  
**Data:** VECTOR, MATRIX, INDEX, SLICE, CONCATENATE, REDUCE, MAP, AGGREGATE, MERGE, ROUTE, TRANSFORM  

All operators have defined arity, type constraints, and real execution semantics.

## Runtime Model

1. Source → AST (parser)
2. AST → Rooted typed computational graph (compiler)
3. Graph is validated (acyclic, reachable from output)
4. Runtime evaluates nodes in topological order
5. Intermediate values are cached; no unnecessary recomputation
6. Resource limits: max nodes, execution timeout

## Evolution

Controlled mutations preserve rootedness and acyclicity.  
Mutations insert real operators (e.g. RELU) rather than placeholders.

## Learning Interface (experimental)

See `src/learning.py` for a minimal parameter-update surface.  
This is deliberately modest; NEUROGEN is a language, not a claim of general intelligence.

## Security Notes

- No arbitrary host code execution from NEUROGEN source.
- Operator set is closed.
- Timeouts and node limits are enforced.
- This is language-level restriction, not OS sandboxing.

---

# NEUROGEN Language Specification v0.3 — Cross-Structure Execution

v0.3 extends v0.2.1 without changing any existing syntax or semantics.
All v0.2.1 programs remain valid.

## Named structure outputs

A `STRUCTURE` may declare named outputs with `OUTPUT <name>` inside its body.
Each name must be bound earlier in the body by `INPUT`, `CONST`, or `LET`.

```neurogen
STRUCTURE perception {
    INPUT image : TENSOR
    LET features = NORMALIZE(image)
    OUTPUT features
}
```

A structure with declared outputs can be compiled and executed independently
(see `src/structure.py`), and can participate in pipelines.

## Pipeline declarations

A `PIPELINE` block connects structures. Connections transport actual computed
runtime values, not symbolic references.

```neurogen
PIPELINE main {
    CONNECT perception.features TO reasoning.evidence
    CONNECT reasoning.decision TO evaluation.score
    OUTPUT evaluation.result
}
```

- `CONNECT <structure>.<output> TO <structure>.<input>` — validated at
  compile time: both structures exist, the named output/input exist, types
  are compatible, and the destination input is not already connected.
- `OUTPUT <structure>.<output>` — selects pipeline outputs (one or more).
- Structures need no knowledge of the pipelines they participate in.

## Connection type rules

- Exact dtype match required (FLOAT→FLOAT, VECTOR→VECTOR, MATRIX→MATRIX,
  BOOL→BOOL, …).
- The only promotion allowed is `INT → FLOAT` (consistent with the v0.2.1
  input rule).
- All other mismatches are rejected **at compile time** with a diagnostic
  naming source, destination, and types.
- The engine re-validates actual runtime values before injection
  (defense in depth); a runtime mismatch fails the receiving structure.

## Execution model

- Structures execute in topological dependency order.
- A structure with multiple upstream inputs runs only after all required
  inputs are available.
- Branching (one output → many inputs): **copy-on-fan-out**. Scalar values
  (BOOL/INT/FLOAT) are immutable and shared; VECTOR/MATRIX/TENSOR and OBJECT
  payloads are deep-copied so branches never share writable memory.
- Shared dependencies execute once per pipeline run; results are reused.
- Cycles are rejected at compile time with diagnostics.
- Every participating structure must lie on a path to a declared pipeline
  output, otherwise the pipeline is rejected as invalid.

## Failure semantics

Per-structure status: `success | failed | blocked`.
- A structure that raises fails explicitly; downstream structures are marked
  `blocked`, never silently given zero values.
- Independent branches keep executing when another branch fails.
- The pipeline reports `failed` if any structure failed or any declared
  output is uncomputable.

## Caching

- Intermediate results are cached when safe. Cache key: structure name +
  structure definition hash + input fingerprint + execution settings.
- Changing inputs invalidates cached results (different fingerprint).
- Structures containing a `CONNECT` (memory write) are **never** served from
  cache: the write executes on every run. Documented limitation, tested.

## Tracing

Pipeline execution produces a structured trace: execution id, per-structure
order/status/inputs/outputs (summarized, not full tensor dumps), connections
traversed, cache hits/misses/bypasses, errors with stage identity, per-stage
durations, and per-structure memory dumps.

## Limits (v0.3)

Defaults, configurable via `EngineSettings` / `PipelineLimits`:
max 64 structures, 1024 connections, dependency depth 32 per pipeline;
per-structure timeout 5s and node limit 10000 (inherited from the v0.2
runtime); total pipeline budget 60s.
Timeouts are checked between operators, not inside an operator.
No tensor-element caps and no OS-level isolation (documented limitations).

## Dependencies

NEUROGEN v0.3 requires Python 3.8+ and `numpy` (tested with 1.26.4).
Install: `pip install numpy`. No other dependencies.
