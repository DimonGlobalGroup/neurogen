# NEUROGEN v0.3 — Benchmarks

**Date:** 2026-11-01
**Environment:** Linux-7.0.0-39-generic x86_64, Python 3.12.3, numpy 1.26.4,
4 CPUs. Command: `PYTHONPATH=src python3 benchmark_v03.py`.

Do not compare these numbers with measurements from other environments.

## Compile time (parse + structure compile + pipeline compile)

| Case | mean | median | p95 | n |
|---|---|---|---|---|
| 1-structure chain | 0.252 ms | 0.202 ms | 1.013 ms | 20 |
| 2-structure chain | 0.382 ms | 0.373 ms | 0.510 ms | 20 |
| 5-structure chain | 0.854 ms | 0.842 ms | 1.066 ms | 20 |
| 10-structure chain | 1.641 ms | 1.627 ms | 1.845 ms | 20 |
| 20-structure chain | 3.234 ms | 3.177 ms | 3.488 ms | 20 |
| branching 1→4 | 0.720 ms | 0.708 ms | 0.939 ms | 20 |
| shared dependency (diamond) | 0.615 ms | 0.606 ms | 0.750 ms | 20 |

Compilation scales approximately linearly with structure count.

## Execution time (pipeline engine, cache disabled unless noted)

| Case | mean | median | p95 | n | cache hit rate |
|---|---|---|---|---|---|
| 1-structure chain | 0.054 ms | 0.051 ms | 0.068 ms | 100 | 0.00 |
| 2-structure chain | 0.095 ms | 0.092 ms | 0.114 ms | 100 | 0.00 |
| 5-structure chain | 0.212 ms | 0.206 ms | 0.235 ms | 100 | 0.00 |
| 10-structure chain | 0.391 ms | 0.385 ms | 0.418 ms | 100 | 0.00 |
| 20-structure chain | 0.760 ms | 0.755 ms | 0.815 ms | 100 | 0.00 |
| branching 1→4 | 0.165 ms | 0.161 ms | 0.184 ms | 100 | 0.00 |
| shared dependency (diamond) | 0.145 ms | 0.142 ms | 0.164 ms | 100 | 0.00 |
| 5-chain uncached | 0.211 ms | 0.207 ms | 0.229 ms | 100 | 0.00 |
| 5-chain cached (warmed) | 0.129 ms | 0.124 ms | 0.169 ms | 100 | 1.00 |

Per-structure execution overhead is roughly 35–45 µs. Caching a warmed
5-structure chain reduces mean execution latency by ~39% (0.211 ms →
0.129 ms) at a 100% hit rate. No failures observed in any benchmark run.

## Memory

Peak RSS delta across the full benchmark session: +256 KB (34,088 KB →
34,344 KB). Pipeline IRs and caches are small; no per-run growth was
observed.

## Notes

- No performance improvement over v0.2.1 is claimed; v0.3 adds pipeline
  orchestration overhead (~40 µs per structure) in exchange for the new
  capability. The v0.2.1 single-structure path is untouched and its
  microbenchmarks are unchanged.
- Copy-on-fan-out adds one deep copy per array-valued branch edge; the
  branching benchmark above includes that cost.
