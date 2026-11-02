#!/usr/bin/env python3
"""NEUROGEN v0.3 — Benchmarks.

Compilation and execution measured separately. Reports mean/median/p95
latency, cache hit rate, and peak RSS where measurable. Environment is
recorded alongside the numbers; do not compare across environments.
"""

import sys, os, time, statistics, platform
import resource
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interface import NeurogenPipeline
from pipeline_engine import EngineSettings


def chain_source(n):
    parts = ['GENOME "Bench"']
    for i in range(n):
        parts.append(f"""STRUCTURE s{i} {{
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    LET z = MULTIPLY(y, 2.0)
    OUTPUT z
}}""")
    conns = [f"    CONNECT s{i}.z TO s{i+1}.x" for i in range(n - 1)]
    conns.append(f"    OUTPUT s{n-1}.z")
    parts.append("PIPELINE main {\n" + "\n".join(conns) + "\n}")
    return "\n".join(parts)


def stats(times):
    s = sorted(times)
    n = len(s)
    p95 = s[int(0.95 * n)] if n else 0.0
    return {
        "n": n,
        "mean_ms": statistics.mean(s) * 1000 if s else 0.0,
        "median_ms": statistics.median(s) * 1000 if s else 0.0,
        "p95_ms": p95 * 1000,
    }


def bench_compile(name, src, n=20):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        NeurogenPipeline().load(src)
        times.append(time.perf_counter() - t0)
    st = stats(times)
    print(f"compile  {name:22s} " + " ".join(f"{k}={v:.3f}" for k, v in st.items() if k != "n") + f"  n={n}")
    return st


def bench_exec(name, src, inputs, n=100, settings=None, warmup=True):
    ngp = NeurogenPipeline().load(src)
    if warmup:
        ngp.run("main", inputs, settings=settings)
    times = []
    hits = 0
    total_struct_runs = 0
    for _ in range(n):
        t0 = time.perf_counter()
        r = ngp.run("main", inputs, settings=settings)
        times.append(time.perf_counter() - t0)
        assert r.status == "success", r.errors
        hits += sum(1 for t in r.trace if t.cache == "hit")
        total_struct_runs += len(r.trace)
    st = stats(times)
    hit_rate = hits / total_struct_runs if total_struct_runs else 0.0
    print(f"exec     {name:22s} " + " ".join(f"{k}={v:.3f}" for k, v in st.items() if k != "n")
          + f"  n={n} hit_rate={hit_rate:.2f}")
    return st


def main():
    print("=" * 78)
    print("NEUROGEN v0.3 benchmarks")
    print(f"platform={platform.platform()} python={platform.python_version()} "
          f"cpu={os.cpu_count()}")
    try:
        import numpy
        print(f"numpy={numpy.__version__}")
    except ImportError:
        print("numpy=missing")
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"baseline peak RSS={rss0} KB")
    print("=" * 78)

    no_cache = EngineSettings(cache_enabled=False)

    for n in (1, 2, 5, 10, 20):
        src = chain_source(n)
        bench_compile(f"{n}-structure chain", src)
        bench_exec(f"{n}-structure chain", src, {"s0.x": 1.0},
                   settings=no_cache)

    branch_src = '''
GENOME "BenchBranch"
STRUCTURE src {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
STRUCTURE b1 {
    INPUT p : FLOAT
    LET r = MULTIPLY(p, 2.0)
    OUTPUT r
}
STRUCTURE b2 {
    INPUT p : FLOAT
    LET r = MULTIPLY(p, 3.0)
    OUTPUT r
}
STRUCTURE b3 {
    INPUT p : FLOAT
    LET r = MULTIPLY(p, 4.0)
    OUTPUT r
}
STRUCTURE b4 {
    INPUT p : FLOAT
    LET r = MULTIPLY(p, 5.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT src.y TO b1.p
    CONNECT src.y TO b2.p
    CONNECT src.y TO b3.p
    CONNECT src.y TO b4.p
    OUTPUT b1.r
    OUTPUT b2.r
    OUTPUT b3.r
    OUTPUT b4.r
}
'''
    bench_compile("branching 1->4", branch_src)
    bench_exec("branching 1->4", branch_src, {"src.x": 1.0}, settings=no_cache)

    shared_src = '''
GENOME "BenchShared"
STRUCTURE root {
    INPUT x : FLOAT
    LET y = POWER(x, 2.0)
    OUTPUT y
}
STRUCTURE l {
    INPUT a : FLOAT
    LET r = ADD(a, 1.0)
    OUTPUT r
}
STRUCTURE r2 {
    INPUT b : FLOAT
    LET s = MULTIPLY(b, 2.0)
    OUTPUT s
}
STRUCTURE join {
    INPUT p : FLOAT
    INPUT q : FLOAT
    LET t = ADD(p, q)
    OUTPUT t
}
PIPELINE main {
    CONNECT root.y TO l.a
    CONNECT root.y TO r2.b
    CONNECT l.r TO join.p
    CONNECT r2.s TO join.q
    OUTPUT join.t
}
'''
    bench_compile("shared dependency", shared_src)
    bench_exec("shared dependency", shared_src, {"root.x": 3.0}, settings=no_cache)

    print("--- cached vs uncached (5-structure chain) ---")
    src5 = chain_source(5)
    bench_exec("5-chain uncached", src5, {"s0.x": 1.0}, settings=no_cache)
    bench_exec("5-chain cached", src5, {"s0.x": 1.0},
               settings=EngineSettings(cache_enabled=True))

    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("=" * 78)
    print(f"peak RSS after benchmarks={rss1} KB (delta={rss1 - rss0} KB)")
    print("=" * 78)


if __name__ == "__main__":
    main()
