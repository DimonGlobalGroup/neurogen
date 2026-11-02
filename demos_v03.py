#!/usr/bin/env python3
"""NEUROGEN v0.3 — Cross-structure pipeline demonstrations.

Every demo is a real .ng program executed through the actual NEUROGEN
compiler and pipeline runtime. Results are verified against independently
computed expectations, never hardcoded in place of execution.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interface import NeurogenPipeline
from pipeline_engine import EngineSettings
from errors import ConnectionError


def load(name):
    path = os.path.join(os.path.dirname(__file__), "examples", "pipelines", name)
    return NeurogenPipeline().load(open(path).read())


def demo1():
    ngp = load("demo1_basic.ng")
    r = ngp.run("main", {"stage_a.x": 5.0})
    assert r.status == "success", r.errors
    # ((5*2)+10)*3 = 60, computed by the runtime
    assert r.outputs["stage_c.s"] == 60.0, r.outputs
    print(f"demo1 basic pipeline: stage_c.s = {r.outputs['stage_c.s']}  OK")


def demo2():
    ngp = load("demo2_multi_input.ng")
    r = ngp.run("main", {"source_left.a": 3.0, "source_right.b": 4.0})
    assert r.status == "success", r.errors
    # (3*2) + (4+1) = 11
    assert r.outputs["combiner.total"] == 11.0, r.outputs
    print(f"demo2 multi-input: combiner.total = {r.outputs['combiner.total']}  OK")


def demo3():
    ngp = load("demo3_vector.ng")
    r = ngp.run("main", {})
    assert r.status == "success", r.errors
    # independent oracle with numpy (not the NEUROGEN runtime)
    v = np.array([1.0, -2.0, 3.0, -4.0])
    n = (v - v.mean()) / (v.std() + 1e-8)
    expected_vec = np.maximum(0, n)
    expected_sum = float(expected_vec.sum())
    got_vec = np.asarray(r.outputs["transform.r"])
    assert np.allclose(got_vec, expected_vec), (got_vec, expected_vec)
    assert abs(r.outputs["evaluate.s"] - expected_sum) < 1e-9
    print(f"demo3 vector pipeline: evaluate.s = {r.outputs['evaluate.s']:.6f}  OK")


def demo4():
    ngp = load("demo4_branch.ng")
    r = ngp.run("main", {"src.x": 2.0})
    assert r.status == "success", r.errors
    # src.y = 5; branch_one = 50; branch_two = 105
    assert r.outputs["branch_one.r"] == 50.0, r.outputs
    assert r.outputs["branch_two.r"] == 105.0, r.outputs
    print(f"demo4 branching: {r.outputs}  OK")


def demo5():
    ngp = load("demo5_shared.ng")
    r = ngp.run("main", {"root.x": 4.0})
    assert r.status == "success", r.errors
    # root.y = 16; left = 17; right = 32
    assert r.outputs["left.r"] == 17.0, r.outputs
    assert r.outputs["right.r"] == 32.0, r.outputs
    # shared upstream executed exactly once in one run
    root_runs = [t for t in r.trace if t.name == "root"]
    assert len(root_runs) == 1, [t.name for t in r.trace]
    # second identical run: everything served from cache
    r2 = ngp.run("main", {"root.x": 4.0})
    assert all(t.cache == "hit" for t in r2.trace), \
        [(t.name, t.cache) for t in r2.trace]
    assert r2.outputs == r.outputs
    print(f"demo5 shared dependency: {r.outputs}, root ran once, "
          f"run2 all cache hits  OK")


def demo6():
    # 6a: compile-time type mismatch is rejected before execution
    bad_src = '''
GENOME "BadConn"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 2.0)
    OUTPUT y
}
STRUCTURE b {
    INPUT v : VECTOR
    LET s = REDUCE(v)
    OUTPUT s
}
PIPELINE main {
    CONNECT a.y TO b.v
    OUTPUT b.s
}
'''
    try:
        NeurogenPipeline().load(bad_src)
        raise AssertionError("type mismatch was NOT rejected at compile time")
    except ConnectionError as e:
        print(f"demo6a compile-time rejection: {e}")

    # 6b: runtime failure -> branch fails, join blocked, independent completes
    ngp = load("demo6_failure.ng")
    r = ngp.run("main", {"good.x": 3.0, "bad.x": 1.0, "independent.v": 5.0})
    assert r.status == "failed", "pipeline must report failure"
    st = r.structure_status
    assert st["bad"] == "failed", st
    assert st["join"] == "blocked", st
    assert st["good"] == "success", st
    assert st["independent"] == "success", st
    assert r.outputs["independent.w"] == 1005.0, r.outputs
    assert "join.r" not in r.outputs, r.outputs
    bad_err = [t.error for t in r.trace if t.name == "bad"][0]
    assert "zero" in bad_err["message"].lower(), bad_err
    print(f"demo6b runtime failure: statuses={st}, "
          f"independent.w={r.outputs['independent.w']}  OK")


def main():
    print("=" * 70)
    print("NEUROGEN v0.3 — Pipeline demonstrations")
    print("=" * 70)
    demo1()
    demo2()
    demo3()
    demo4()
    demo5()
    demo6()
    print("=" * 70)
    print("All six v0.3 demonstrations executed and verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
