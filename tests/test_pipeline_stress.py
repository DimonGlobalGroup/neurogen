"""
NEUROGEN v0.3 — Stress test suite (seeded RNG, reproducible).

- 5,000 randomized valid pipeline executions
- 5,000 randomized invalid-connection tests
- 1,000 randomized multi-structure graphs
- 1,000 repeated executions with changing inputs
- 1,000 cache invalidation scenarios
"""

import sys, os, random, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse
from pipeline import PipelineCompiler
from pipeline_engine import EngineSettings
from interface import NeurogenPipeline
from ast_nodes import Pipeline
from errors import ConnectionError, PipelineError

OPS = ["ADD", "SUBTRACT", "MULTIPLY"]


def gen_structure(rng, name, n_inputs):
    """One FLOAT structure with n_inputs pipeline-fed inputs, random LETs."""
    in_names = [f"in{i}" for i in range(n_inputs)]
    avail = list(in_names)
    lines = [f"STRUCTURE {name} {{"]
    for inp in in_names:
        lines.append(f"    INPUT {inp} : FLOAT")
    for i in range(rng.randint(1, 3)):
        op = rng.choice(OPS)
        a = rng.choice(avail + [round(rng.uniform(-3, 3), 2)])
        b = rng.choice(avail + [round(rng.uniform(-3, 3), 2)])
        t = f"t{i}"
        lines.append(f"    LET {t} = {op}({a}, {b})")
        avail.append(t)
    out = rng.choice(avail)
    lines.append(f"    OUTPUT {out}")
    lines.append("}")
    return "\n".join(lines), in_names, out


def gen_pipeline_source(rng, n_min=2, n_max=4):
    """Random valid pipeline source. Returns (source, entry_inputs)."""
    n = rng.randint(n_min, n_max)
    names = [f"s{i}" for i in range(n)]
    edges = []
    for j in range(1, n):
        i = rng.randint(0, j - 1)
        edges.append((i, j))
        for i2 in range(j):
            if i2 != i and rng.random() < 0.25:
                edges.append((i2, j))
    incoming = {j: [] for j in range(n)}
    outgoing = {j: [] for j in range(n)}
    for i, j in edges:
        incoming[j].append(i)
        outgoing[i].append(j)

    parts = ['GENOME "Stress"']
    struct_out = {}
    struct_inputs = {}
    for j in range(n):
        n_in = max(len(incoming[j]), 1 if not incoming[j] else 0)
        if not incoming[j]:
            n_in = rng.randint(1, 2)
        else:
            n_in = len(incoming[j])
        src, in_names, out = gen_structure(rng, names[j], n_in)
        parts.append(src)
        struct_out[names[j]] = out
        struct_inputs[names[j]] = in_names

    conn_lines = []
    for j in range(n):
        for k, i in enumerate(incoming[j]):
            conn_lines.append(
                f"    CONNECT {names[i]}.{struct_out[names[i]]} "
                f"TO {names[j]}.{struct_inputs[names[j]][k]}")
    sinks = [names[j] for j in range(n) if not outgoing[j]]
    out_lines = [f"    OUTPUT {s}.{struct_out[s]}" for s in sinks]
    parts.append("PIPELINE main {\n" + "\n".join(conn_lines + out_lines) + "\n}")

    # pipeline-fed inputs: every input not targeted by a connection
    fed = set()
    for j in range(n):
        for k in range(len(incoming[j])):
            fed.add((names[j], struct_inputs[names[j]][k]))
    entry_inputs = {}
    for j in range(n):
        for inp in struct_inputs[names[j]]:
            if (names[j], inp) not in fed:
                entry_inputs[f"{names[j]}.{inp}"] = rng.uniform(-5, 5)
    return "\n".join(parts), entry_inputs


def test_valid_pipeline_executions(n=5000, seed=20261101):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    fails = 0
    for i in range(n):
        src, inputs = gen_pipeline_source(rng)
        try:
            ngp = NeurogenPipeline().load(src)
            r = ngp.run("main", inputs)
            assert r.status == "success", r.errors
            for v in r.outputs.values():
                assert isinstance(v, float) and math.isfinite(v), v
        except Exception as e:
            fails += 1
            if fails <= 3:
                print(f"  iteration {i} failed: {type(e).__name__}: {e}")
    dur = time.perf_counter() - t0
    print(f"valid pipeline executions: {n - fails}/{n} passed in {dur:.1f}s "
          f"(seed={seed})")
    assert fails == 0, f"{fails} failures"


INVALID_GENOME = '''
GENOME "Inv"
STRUCTURE fa {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
STRUCTURE fb {
    INPUT z : FLOAT
    LET r = MULTIPLY(z, 2.0)
    OUTPUT r
}
STRUCTURE va {
    INPUT t : VECTOR
    LET n = NORMALIZE(t)
    OUTPUT n
}
STRUCTURE vb {
    INPUT u : VECTOR
    LET s = REDUCE(u)
    OUTPUT s
}
STRUCTURE ia {
    INPUT k : INT
    OUTPUT k
}
'''

INVALID_TEMPLATES = [
    ("typemismatch", "CONNECT fa.y TO vb.u", "OUTPUT fb.r", ConnectionError),
    ("typemismatch", "CONNECT va.n TO fb.z", "OUTPUT fb.r", ConnectionError),
    ("typemismatch", "CONNECT fa.y TO ia.k", "OUTPUT ia.k", ConnectionError),
    ("unknown_struct", "CONNECT fa.y TO zz.z", "OUTPUT fb.r", ConnectionError),
    ("unknown_output", "CONNECT fa.zzz TO fb.z", "OUTPUT fb.r", ConnectionError),
    ("unknown_input", "CONNECT fa.y TO fb.zzz", "OUTPUT fb.r", ConnectionError),
    ("duplicate", "CONNECT fa.y TO fb.z\n    CONNECT fa.y TO fb.z",
     "OUTPUT fb.r", ConnectionError),
    ("cycle", "CONNECT fa.y TO fb.z\n    CONNECT fb.r TO fa.x",
     "OUTPUT fb.r", PipelineError),
    ("selfloop", "CONNECT fa.y TO fa.x", "OUTPUT fa.y", PipelineError),
]


def test_invalid_connection_stress(n=5000, seed=20261102):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    cats = {}
    wrong = 0
    for _ in range(n):
        cat, conn, out, errcls = rng.choice(INVALID_TEMPLATES)
        src = INVALID_GENOME + f"PIPELINE m {{\n    {conn}\n    {out}\n}}\n"
        try:
            g = parse(src)
            p = [s for s in g.statements if isinstance(s, Pipeline)][0]
            PipelineCompiler(g).compile(p)
            wrong += 1
            print(f"  ACCEPTED invalid [{cat}]: {conn}")
        except errcls:
            cats[cat] = cats.get(cat, 0) + 1
        except Exception as e:
            wrong += 1
            print(f"  WRONG CLASS [{cat}]: {type(e).__name__}: {e}")
    dur = time.perf_counter() - t0
    print(f"invalid connections: {n - wrong}/{n} correctly rejected "
          f"in {dur:.1f}s (seed={seed}); categories={cats}")
    assert wrong == 0, f"{wrong} wrong/accepted"


def test_multi_structure_graphs(n=1000, seed=20261103):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    fails = 0
    for i in range(n):
        src, inputs = gen_pipeline_source(rng, n_min=3, n_max=8)
        try:
            ngp = NeurogenPipeline().load(src)
            r = ngp.run("main", inputs)
            assert r.status == "success", r.errors
            assert len(r.trace) >= 3
        except Exception as e:
            fails += 1
            if fails <= 3:
                print(f"  iteration {i} failed: {type(e).__name__}: {e}")
    dur = time.perf_counter() - t0
    print(f"multi-structure graphs: {n - fails}/{n} passed in {dur:.1f}s "
          f"(seed={seed})")
    assert fails == 0


DIAMOND = '''
GENOME "Rep"
STRUCTURE top {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 2.0)
    OUTPUT y
}
STRUCTURE left {
    INPUT a : FLOAT
    LET r = ADD(a, 10.0)
    OUTPUT r
}
STRUCTURE right {
    INPUT b : FLOAT
    LET r = MULTIPLY(b, 3.0)
    OUTPUT r
}
STRUCTURE join {
    INPUT p : FLOAT
    INPUT q : FLOAT
    LET r = ADD(p, q)
    OUTPUT r
}
PIPELINE main {
    CONNECT top.y TO left.a
    CONNECT top.y TO right.b
    CONNECT left.r TO join.p
    CONNECT right.r TO join.q
    OUTPUT join.r
}
'''


def test_repeated_changing_inputs(n=1000, seed=20261104):
    rng = random.Random(seed)
    ngp = NeurogenPipeline().load(DIAMOND)
    t0 = time.perf_counter()
    fails = 0
    for i in range(n):
        x = rng.uniform(-10, 10)
        r = ngp.run("main", {"top.x": x})
        expected = (2 * x + 10) + (2 * x * 3)  # = 8x + 10
        if r.status != "success" or abs(r.outputs["join.r"] - expected) > 1e-9:
            fails += 1
            if fails <= 3:
                print(f"  iteration {i}: x={x} got={r.outputs} status={r.status}")
        if i % 100 == 0:  # determinism spot check
            r2 = ngp.run("main", {"top.x": x})
            assert r2.outputs == r.outputs
    dur = time.perf_counter() - t0
    print(f"repeated changing inputs: {n - fails}/{n} passed in {dur:.1f}s "
          f"(seed={seed})")
    assert fails == 0


def test_cache_invalidation(n=1000, seed=20261105):
    rng = random.Random(seed)
    src = '''
GENOME "CI"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
STRUCTURE b {
    INPUT z : FLOAT
    LET r = MULTIPLY(z, 2.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.y TO b.z
    OUTPUT b.r
}
'''
    ngp = NeurogenPipeline().load(src)
    t0 = time.perf_counter()
    fails = 0
    for _ in range(n):
        x0 = rng.uniform(-5, 5)
        r1 = ngp.run("main", {"a.x": x0})
        r2 = ngp.run("main", {"a.x": x0})
        x1 = x0 + rng.uniform(0.5, 2.0)
        r3 = ngp.run("main", {"a.x": x1})
        r4 = ngp.run("main", {"a.x": x1})
        c1 = [t.cache for t in r1.trace]
        c2 = [t.cache for t in r2.trace]
        c3 = [t.cache for t in r3.trace]
        c4 = [t.cache for t in r4.trace]
        ok = (c1 == ["miss", "miss"] and c2 == ["hit", "hit"]
              and c3 == ["miss", "miss"] and c4 == ["hit", "hit"]
              and abs(r3.outputs["b.r"] - 2 * (x1 + 1)) < 1e-9
              and r2.outputs == r1.outputs)
        if not ok:
            fails += 1
            if fails <= 3:
                print(f"  failed: {c1} {c2} {c3} {c4} {r3.outputs}")
    dur = time.perf_counter() - t0
    print(f"cache invalidation: {n - fails}/{n} passed in {dur:.1f}s "
          f"(seed={seed})")
    assert fails == 0


def run_all():
    test_valid_pipeline_executions()
    test_invalid_connection_stress()
    test_multi_structure_graphs()
    test_repeated_changing_inputs()
    test_cache_invalidation()
    print("\nAll stress suites passed.")


if __name__ == "__main__":
    run_all()
