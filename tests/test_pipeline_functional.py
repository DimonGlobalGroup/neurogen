"""
NEUROGEN v0.3 — Functional test suite for cross-structure pipelines.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse
from structure import compile_structure, infer_node_dtype
from pipeline import PipelineCompiler
from pipeline_engine import PipelineEngine, EngineSettings, fanout_value
from interface import NeurogenPipeline, Neurogen
from neuro_types import NeuroValue, DType
from errors import (StructureError, ConnectionError, PipelineError,
                    StructureInputError, StructureExecutionError)


def _load_pipeline(src):
    return NeurogenPipeline().load(src)


def test_single_structure_independent_new_interface():
    """Refinement 4: compile and run a single STRUCTURE via the new interface."""
    src = '''
GENOME "S"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 2.0)
    OUTPUT y
}
'''
    ngp = NeurogenPipeline().load(src)
    r = ngp.run_structure("a", {"x": 21.0})
    assert r["outputs"]["y"] == 42.0, r
    assert r["result"] == 42.0


def test_single_structure_legacy_path():
    """Refinement 4: the same structure via the legacy genome-level OUTPUT path."""
    src = '''
GENOME "S"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 2.0)
}
OUTPUT a
'''
    ng = Neurogen().load(src)
    assert ng.run({"x": 21.0}) == 42.0


def test_two_structure_pipeline():
    src = '''
GENOME "T"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
STRUCTURE b {
    INPUT z : FLOAT
    LET r = MULTIPLY(z, 10.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.y TO b.z
    OUTPUT b.r
}
'''
    r = _load_pipeline(src).run("main", {"a.x": 4.0})
    assert r.status == "success", r.errors
    assert r.outputs["b.r"] == 50.0  # (4+1)*10


def test_multi_stage_pipeline():
    src = '''
GENOME "M"
STRUCTURE s1 {
    INPUT x : FLOAT
    LET y = ADD(x, 2.0)
    OUTPUT y
}
STRUCTURE s2 {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 3.0)
    OUTPUT y
}
STRUCTURE s3 {
    INPUT x : FLOAT
    LET y = SUBTRACT(x, 1.0)
    OUTPUT y
}
STRUCTURE s4 {
    INPUT x : FLOAT
    LET y = POWER(x, 2.0)
    OUTPUT y
}
PIPELINE main {
    CONNECT s1.y TO s2.x
    CONNECT s2.y TO s3.x
    CONNECT s3.y TO s4.x
    OUTPUT s4.y
}
'''
    r = _load_pipeline(src).run("main", {"s1.x": 1.0})
    assert r.status == "success", r.errors
    assert r.outputs["s4.y"] == ((1.0 + 2.0) * 3.0 - 1.0) ** 2.0
    assert [t.name for t in r.trace] == ["s1", "s2", "s3", "s4"]


def test_three_upstream_inputs():
    src = '''
GENOME "U"
STRUCTURE p { INPUT x : FLOAT
    LET y = MULTIPLY(x, 2.0)
    OUTPUT y }
STRUCTURE q { INPUT x : FLOAT
    LET y = MULTIPLY(x, 3.0)
    OUTPUT y }
STRUCTURE w { INPUT x : FLOAT
    LET y = MULTIPLY(x, 4.0)
    OUTPUT y }
STRUCTURE comb { INPUT a : FLOAT
    INPUT b : FLOAT
    INPUT c : FLOAT
    LET t = ADD(a, b)
    LET total = ADD(t, c)
    OUTPUT total }
PIPELINE main {
    CONNECT p.y TO comb.a
    CONNECT q.y TO comb.b
    CONNECT w.y TO comb.c
    OUTPUT comb.total
}
'''
    r = _load_pipeline(src).run("main", {"p.x": 1.0, "q.x": 1.0, "w.x": 1.0})
    assert r.status == "success", r.errors
    assert r.outputs["comb.total"] == 2.0 + 3.0 + 4.0


def test_multiple_outputs():
    src = '''
GENOME "MO"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    LET z = MULTIPLY(x, 2.0)
    OUTPUT y
    OUTPUT z
}
STRUCTURE b {
    INPUT p : FLOAT
    INPUT q : FLOAT
    LET r = ADD(p, q)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.y TO b.p
    CONNECT a.z TO b.q
    OUTPUT b.r
    OUTPUT a.z
}
'''
    r = _load_pipeline(src).run("main", {"a.x": 5.0})
    assert r.status == "success", r.errors
    assert r.outputs["b.r"] == 6.0 + 10.0
    assert r.outputs["a.z"] == 10.0


def test_branching_values_correct():
    src = '''
GENOME "B"
STRUCTURE src {
    INPUT x : FLOAT
    LET y = ADD(x, 5.0)
    OUTPUT y
}
STRUCTURE b1 {
    INPUT p : FLOAT
    LET r = MULTIPLY(p, 2.0)
    OUTPUT r
}
STRUCTURE b2 {
    INPUT q : FLOAT
    LET r = POWER(q, 2.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT src.y TO b1.p
    CONNECT src.y TO b2.q
    OUTPUT b1.r
    OUTPUT b2.r
}
'''
    r = _load_pipeline(src).run("main", {"src.x": 3.0})
    assert r.status == "success", r.errors
    assert r.outputs["b1.r"] == 16.0   # (3+5)*2
    assert r.outputs["b2.r"] == 64.0   # (3+5)^2


def test_fanout_no_shared_writable_memory():
    """Refinement 1: numpy arrays must not share writable memory between branches."""
    arr = np.array([1.0, 2.0, 3.0])
    v = NeuroValue(arr, DType.VECTOR, arr.shape)
    c1 = fanout_value(v)
    c2 = fanout_value(v)
    assert not np.shares_memory(c1.data, c2.data)
    assert not np.shares_memory(c1.data, v.data)
    c1.data[0] = 999.0
    assert c2.data[0] == 1.0 and v.data[0] == 1.0
    # scalars are immutable: sharing is allowed
    s = NeuroValue(3.14, DType.FLOAT)
    assert fanout_value(s) is s
    b = NeuroValue(True, DType.BOOL)
    assert fanout_value(b) is b
    # OBJECT payloads are deep-copied
    o = NeuroValue({"a": [1, 2]}, DType.OBJECT)
    oc = fanout_value(o)
    assert oc is not o and oc.data == o.data
    print("✓ fanout_no_shared_writable_memory")


def test_fanout_end_to_end_vector():
    """Branching a VECTOR: both branches compute correctly (no corruption)."""
    src = '''
GENOME "FB"
STRUCTURE src {
    INPUT v : VECTOR
    OUTPUT v
}
STRUCTURE b1 {
    INPUT p : VECTOR
    LET s = REDUCE(p)
    OUTPUT s
}
STRUCTURE b2 {
    INPUT q : VECTOR
    LET n = NORMALIZE(q)
    LET s = REDUCE(n)
    OUTPUT s
}
PIPELINE main {
    CONNECT src.v TO b1.p
    CONNECT src.v TO b2.q
    OUTPUT b1.s
    OUTPUT b2.s
}
'''
    r = _load_pipeline(src).run("main", {"src.v": [1.0, 2.0, 3.0, 4.0]})
    assert r.status == "success", r.errors
    assert abs(r.outputs["b1.s"] - 10.0) < 1e-9
    assert abs(r.outputs["b2.s"] - 0.0) < 1e-9  # normalized values sum to ~0


def test_shared_dependency_executes_once():
    src = '''
GENOME "SH"
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
    LET r = SUBTRACT(b, 1.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT root.y TO l.a
    CONNECT root.y TO r2.b
    OUTPUT l.r
    OUTPUT r2.r
}
'''
    ngp = _load_pipeline(src)
    r = ngp.run("main", {"root.x": 5.0})
    assert r.status == "success", r.errors
    assert r.outputs["l.r"] == 26.0 and r.outputs["r2.r"] == 24.0
    assert len([t for t in r.trace if t.name == "root"]) == 1


def test_dtype_connections():
    """All same-dtype connections validate and execute."""
    for dtype, pyval in [("BOOL", True), ("INT", 7), ("FLOAT", 2.5),
                         ("VECTOR", [1.0, 2.0]), ("MATRIX", [[1.0, 2.0], [3.0, 4.0]])]:
        src = f'''
GENOME "DT"
STRUCTURE a {{
    INPUT x : {dtype}
    OUTPUT x
}}
STRUCTURE b {{
    INPUT y : {dtype}
    OUTPUT y
}}
PIPELINE main {{
    CONNECT a.x TO b.y
    OUTPUT b.y
}}
'''
        r = _load_pipeline(src).run("main", {"a.x": pyval})
        assert r.status == "success", (dtype, r.errors)


def test_int_to_float_promotion():
    src = '''
GENOME "PR"
STRUCTURE a {
    INPUT x : INT
    OUTPUT x
}
STRUCTURE b {
    INPUT y : FLOAT
    LET r = ADD(y, 0.5)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.x TO b.y
    OUTPUT b.r
}
'''
    r = _load_pipeline(src).run("main", {"a.x": 4})
    assert r.status == "success", r.errors
    assert abs(r.outputs["b.r"] - 4.5) < 1e-9


def test_runtime_defense_in_depth_any():
    """Untyped (ANY) output connected to FLOAT input: compile passes, but a
    vector value at runtime must fail the receiving structure loudly."""
    src = '''
GENOME "DD"
STRUCTURE a {
    INPUT x
    OUTPUT x
}
STRUCTURE b {
    INPUT y : FLOAT
    LET r = ADD(y, 1.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.x TO b.y
    OUTPUT b.r
}
'''
    ngp = _load_pipeline(src)  # compiles: ANY defers to runtime
    r = ngp.run("main", {"a.x": 2.0})
    assert r.status == "success" and r.outputs["b.r"] == 3.0
    r2 = ngp.run("main", {"a.x": [1.0, 2.0]})
    assert r2.status == "failed", r2.status
    assert r2.structure_status["b"] == "failed"
    err = [t.error for t in r2.trace if t.name == "b"][0]
    assert err["class"] == "StructureInputError", err


def test_repeated_execution_deterministic():
    src = '''
GENOME "DET"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 1.5)
    OUTPUT y
}
STRUCTURE b {
    INPUT z : FLOAT
    LET r = ADD(z, 2.5)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.y TO b.z
    OUTPUT b.r
}
'''
    ngp = _load_pipeline(src)
    outs = [ngp.run("main", {"a.x": 3.0}).outputs for _ in range(20)]
    assert all(o == outs[0] for o in outs)
    assert outs[0]["b.r"] == 3.0 * 1.5 + 2.5


def test_cache_hit_and_invalidation():
    src = '''
GENOME "CH"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
PIPELINE main {
    OUTPUT a.y
}
'''
    ngp = _load_pipeline(src)
    r1 = ngp.run("main", {"a.x": 1.0})
    assert [t.cache for t in r1.trace] == ["miss"]
    r2 = ngp.run("main", {"a.x": 1.0})
    assert [t.cache for t in r2.trace] == ["hit"]
    assert r2.outputs == r1.outputs
    r3 = ngp.run("main", {"a.x": 2.0})
    assert [t.cache for t in r3.trace] == ["miss"]
    assert r3.outputs["a.y"] == 3.0


def test_cache_disabled():
    src = '''
GENOME "CD"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
PIPELINE main {
    OUTPUT a.y
}
'''
    ngp = _load_pipeline(src)
    s = EngineSettings(cache_enabled=False)
    r1 = ngp.run("main", {"a.x": 1.0}, settings=s)
    r2 = ngp.run("main", {"a.x": 1.0}, settings=s)
    assert [t.cache for t in r1.trace] == ["disabled"]
    assert [t.cache for t in r2.trace] == ["disabled"]
    assert r2.outputs["a.y"] == 2.0


def test_connect_never_cached():
    """Refinement 2: CONNECT structures execute (and write) on every run."""
    src = '''
GENOME "CN"
MEMORY keyvalue
STRUCTURE w {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    CONNECT writes
    OUTPUT y
}
PIPELINE main {
    OUTPUT w.y
}
'''
    ngp = _load_pipeline(src)
    r1 = ngp.run("main", {"w.x": 1.0})
    r2 = ngp.run("main", {"w.x": 1.0})
    t1 = [t for t in r1.trace if t.name == "w"][0]
    t2 = [t for t in r2.trace if t.name == "w"][0]
    assert t1.cache == "bypass" and t2.cache == "bypass", (t1.cache, t2.cache)
    assert t1.memory.get("writes") == 2.0, t1.memory
    assert t2.memory.get("writes") == 2.0, t2.memory
    print("✓ connect_never_cached")


def test_tracing_structure():
    src = '''
GENOME "TR"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 2.0)
    OUTPUT y
}
STRUCTURE b {
    INPUT z : FLOAT
    LET r = ADD(z, 1.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.y TO b.z
    OUTPUT b.r
}
'''
    r = _load_pipeline(src).run("main", {"a.x": 3.0})
    assert r.execution_id and len(r.execution_id) == 12
    assert len(r.trace) == 2
    ta, tb = r.trace
    assert (ta.order, ta.name, ta.status) == (0, "a", "success")
    assert (tb.order, tb.name, tb.status) == (1, "b", "success")
    assert ta.inputs_summary == {"x": "FLOAT(3.0)"}, ta.inputs_summary
    assert ta.outputs_summary == {"y": "FLOAT(6.0)"}, ta.outputs_summary
    assert r.connections_traversed == [("a", "y", "b", "z")]
    assert all(t.duration_s >= 0 for t in r.trace)
    assert r.duration_s >= 0
    # execution ids unique across runs
    r2 = _load_pipeline(src).run("main", {"a.x": 3.0})
    assert r2.execution_id != r.execution_id


def test_failure_propagation_blocks_downstream():
    src = '''
GENOME "FP"
STRUCTURE bad {
    INPUT x : FLOAT
    CONST zero = 0.0
    LET r = DIVIDE(x, zero)
    OUTPUT r
}
STRUCTURE down {
    INPUT q : FLOAT
    LET s = ADD(q, 1.0)
    OUTPUT s
}
PIPELINE main {
    CONNECT bad.r TO down.q
    OUTPUT down.s
}
'''
    r = _load_pipeline(src).run("main", {"bad.x": 5.0})
    assert r.status == "failed"
    assert r.structure_status["bad"] == "failed"
    assert r.structure_status["down"] == "blocked"
    assert r.outputs == {}
    err = [t.error for t in r.trace if t.name == "bad"][0]
    assert err["class"] == "StructureExecutionError", err
    assert "zero" in err["message"].lower()


def test_failure_isolation_diamond():
    """Refinement 3: one failing branch, the other completes, join blocked."""
    src = '''
GENOME "DI"
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
    CONST zero = 0.0
    LET r = DIVIDE(b, zero)
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
    r = _load_pipeline(src).run("main", {"top.x": 4.0})
    st = r.structure_status
    assert st["top"] == "success", st
    assert st["left"] == "success", st
    assert st["right"] == "failed", st
    assert st["join"] == "blocked", st
    assert r.status == "failed"
    print("✓ failure_isolation_diamond")


def test_pipeline_input_forms():
    src = '''
GENOME "IF"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
STRUCTURE b {
    INPUT z : FLOAT
    LET r = ADD(z, 2.0)
    OUTPUT r
}
PIPELINE main {
    OUTPUT a.y
    OUTPUT b.r
}
'''
    ngp = _load_pipeline(src)
    base = {"a.x": 1.0, "b.z": 10.0}
    r1 = ngp.run("main", base)
    r2 = ngp.run("main", {("a", "x"): 1.0, ("b", "z"): 10.0})
    r3 = ngp.run("main", {"x": 1.0, "z": 10.0})  # bare, unambiguous
    assert r1.outputs == r2.outputs == r3.outputs == {"a.y": 2.0, "b.r": 12.0}


def test_ambiguous_bare_input():
    src = '''
GENOME "AM"
STRUCTURE a {
    INPUT x : FLOAT
    OUTPUT x
}
STRUCTURE b {
    INPUT x : FLOAT
    OUTPUT x
}
PIPELINE main {
    OUTPUT a.x
    OUTPUT b.x
}
'''
    ngp = _load_pipeline(src)
    try:
        from pipeline_engine import PipelineEngine
        eng = PipelineEngine(ngp.pipelines["main"])
        eng.execute({"x": 1.0})
        raise AssertionError("ambiguous input accepted")
    except StructureInputError as e:
        assert "mbiguous" in str(e), str(e)


def test_inspect_pipeline():
    src = '''
GENOME "IN"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
STRUCTURE b {
    INPUT z : FLOAT
    LET r = ADD(z, 2.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.y TO b.z
    OUTPUT b.r
}
'''
    info = _load_pipeline(src).inspect_pipeline("main")
    assert info["exec_order"] == ["a", "b"]
    assert info["connections"] == [
        {"from": "a.y", "to": "b.z", "types": "FLOAT->FLOAT"}]
    assert info["structures"]["a"]["outputs"] == {"y": "FLOAT"}


def run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for fn in fns:
        try:
            fn()
            print(f"✓ {fn.__name__}")
            passed += 1
        except Exception:
            print(f"✗ {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\nfunctional: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
