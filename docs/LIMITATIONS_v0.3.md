# NEUROGEN v0.3 — Known Limitations

These limitations are explicit. Do not represent v0.3 as having solved them.

## Carried over from v0.2.1 (unchanged)

1. **Memory is in-process.** Associative and key-value memory do not survive
   process restarts. Pipelines do not add durability.
2. **No intelligent learning.** `src/learning.py` is a numerical-gradient
   helper for parameter tweaking, not autonomous learning. Data moving
   through pipelines is not learning.
3. **Mutation quality not established.** `ControlledEvolver` preserves
   structural validity only; whether mutations improve behavior is unmeasured.
   Mutation preserves the graph's `output_id` root; multi-output structures
   should be mutated with care (see `reinterface` in `src/structure.py`).
4. **No tensor-size or memory budgets.** Node-count limits and timeouts
   exist; there are no caps on tensor elements or intermediate memory.
5. **Timeout granularity.** Timeouts are checked between operators, not
   inside an individual operator. A single hung operator cannot be
   interrupted by the timeout.
6. **No neural-network integration.** No model adapters, inference nodes,
   or training interfaces were added.
7. **No general runtime self-modification.** Controlled graph mutation is
   not architecture self-modification.
8. **No OS-level isolation.** Restrictions are language-level (closed
   operator set, timeouts, node limits). Do not run untrusted NEUROGEN
   programs expecting sandboxing.

## v0.3 boundaries (new)

9. **Cycles rejected, not iterated.** Circular pipeline dependencies are a
   compile-time error. Recurrent / loopy execution is not supported.
10. **Type promotion is INT→FLOAT only.** All other cross-type connections
    are rejected at compile time. `ANY`-typed (undeclared) values defer to
    runtime validation.
11. **Static output-type inference is best-effort.** The compiler
    approximates operator result types for connection validation; imprecise
    cases degrade to runtime checks, never to silent acceptance.
12. **Side-effect structures are cache-exempt but still single-run.**
    A `CONNECT` structure executes once per pipeline run; there is no
    cross-run memory persistence.
13. **Every pipeline structure must reach a declared output.** Pure
    side-effect sinks with no path to an output are rejected as unreachable.
14. **Trace summaries, not dumps.** Tensor contents are summarized as
    `DTYPE[shape]`; full values are available via `to_python()` on
    `PipelineResult.outputs` only for declared pipeline outputs.
15. **numpy is a hard dependency.** `neuro_types.py` and `operators.py`
    require numpy (tested with 1.26.4).

## What v0.3 does establish

Independent NEUROGEN structures with declared typed interfaces can exchange
real computed values through validated pipelines, with deterministic
scheduling, inspectable tracing, safe caching, and explicit failure
semantics — measured by 13,087 executed checks, all passing.
