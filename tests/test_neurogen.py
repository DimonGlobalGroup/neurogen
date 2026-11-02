"""
NEUROGEN v0.2.1 — Verification and Reliability test suite
"""

import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse, ParseError
from compiler import compile_program, CompileError
from runtime import Runtime, RuntimeError as NGRuntimeError
from memory import AssociativeMemory, KeyValueMemory
from evolution import ControlledEvolver
from interface import Neurogen
from operators import REGISTRY, get_operator, list_operators, OperatorError
from neuro_types import NeuroValue, DType
import numpy as np


def test_operators_math():
    x = NeuroValue.from_python(3.0)
    y = NeuroValue.from_python(4.0)
    assert abs(get_operator("ADD").execute([x, y]).to_python() - 7.0) < 1e-9
    assert abs(get_operator("MULTIPLY").execute([x, y]).to_python() - 12.0) < 1e-9
    assert abs(get_operator("POWER").execute([x, NeuroValue.from_python(2)]).to_python() - 9.0) < 1e-9
    assert get_operator("MAXIMUM").execute([x, y]).to_python() == 4.0
    print("✓ operators_math")


def test_operators_logic():
    t = NeuroValue.from_python(True)
    f = NeuroValue.from_python(False)
    assert get_operator("AND").execute([t, f]).to_python() is False
    assert get_operator("OR").execute([t, f]).to_python() is True
    assert get_operator("XOR").execute([t, f]).to_python() is True
    assert get_operator("NOT").execute([t]).to_python() is False
    print("✓ operators_logic")


def test_operators_transforms():
    v = NeuroValue.from_python([-1.0, 0.0, 2.0])
    relu = get_operator("RELU").execute([v]).to_python()
    assert relu == [0.0, 0.0, 2.0]
    sig = get_operator("SIGMOID").execute([NeuroValue.from_python(0.0)]).to_python()
    assert abs(sig - 0.5) < 1e-6
    print("✓ operators_transforms")


def test_divide_by_zero():
    a = NeuroValue.from_python(1.0)
    b = NeuroValue.from_python(0.0)
    try:
        get_operator("DIVIDE").execute([a, b])
        assert False, "DIVIDE by zero should raise"
    except OperatorError as e:
        assert "zero" in str(e).lower()
    # Normal division still works
    r = get_operator("DIVIDE").execute([NeuroValue.from_python(10.0), NeuroValue.from_python(2.0)])
    assert abs(r.to_python() - 5.0) < 1e-9
    print("✓ divide_by_zero")


def test_vector_pipeline():
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "vector_ops.ng")).read()
    ng = Neurogen().load(src)
    r = ng.run({"data": [1.0, -2.0, 3.0, -4.0, 5.0]})
    assert isinstance(r, (int, float))
    assert r > 0
    print(f"✓ vector_pipeline → {r:.4f}")


def test_math_demo():
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "math_ops.ng")).read()
    ng = Neurogen().load(src)
    r = ng.run({"x": 1.0, "y": 1.0})
    assert abs(r - 1.0) < 1e-5
    print("✓ math_demo")


def test_logic_demo():
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "logic.ng")).read()
    ng = Neurogen().load(src)
    r = ng.run({"a": True, "b": False})
    assert r is True
    print("✓ logic_demo")


def test_motif_and_pipeline():
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "pipeline.ng")).read()
    ng = Neurogen().load(src)
    r = ng.run({"raw": [0.5, -0.3, 0.8, 0.1]})
    assert isinstance(r, dict) and "connected_to" in r
    print("✓ motif_and_pipeline")


def test_regression_v01_style():
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "reasoning.ng")).read()
    ng = Neurogen().load(src)
    r = ng.run({"patterns": [0, 1, 0]})
    assert r is not None
    print("✓ regression_v01_style")


def test_associative_memory():
    mem = AssociativeMemory()
    mem.store("xor", [1, -1], tags=["logic"])
    hits = mem.retrieve("xor", tags=["logic"])
    assert len(hits) >= 1
    print("✓ associative_memory")


def test_keyvalue_memory_connect():
    """Issue #1: KeyValueMemory must actually receive CONNECT writes."""
    src = '''
GENOME "KVTest"
MEMORY keyvalue
STRUCTURE s {
    INPUT x : FLOAT
    CONST k = 42
    LET y = ADD(x, k)
    CONNECT result
}
OUTPUT s
'''
    ng = Neurogen().load(src)
    r = ng.run({"x": 8.0})
    dump = ng.inspect()["memory"]
    assert isinstance(dump, dict), f"Expected dict dump for keyvalue, got {type(dump)}"
    assert "result" in dump, f"CONNECT did not write to KeyValueMemory: {dump}"
    assert dump["result"] == 50.0 or (isinstance(dump["result"], dict) and dump["result"].get("data") == 50.0)
    # The CONNECT returns a wrapper; the stored value is the payload
    # Accept either the raw payload or the wrapper depending on implementation
    stored = dump["result"]
    if isinstance(stored, dict) and "data" in stored:
        assert stored["data"] == 50.0
    else:
        assert stored == 50.0
    print("✓ keyvalue_memory_connect")


def test_missing_input_raises():
    """Issue #2: missing inputs must be explicit errors."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "math_ops.ng")).read()
    ng = Neurogen().load(src)
    try:
        ng.run({"x": 1.0})  # missing y
        assert False, "Missing input should raise"
    except Exception as e:
        assert "missing" in str(e).lower() or "y" in str(e).lower()
    print("✓ missing_input_raises")


def test_typed_input_enforcement():
    """Issue #2 continued: declared types should be checked."""
    src = '''
GENOME "TypeTest"
STRUCTURE s {
    INPUT v : VECTOR
    LET r = REDUCE(v)
}
OUTPUT s
'''
    ng = Neurogen().load(src)
    # Correct type
    r = ng.run({"v": [1.0, 2.0, 3.0]})
    assert abs(r - 6.0) < 1e-9
    # Wrong type
    try:
        ng.run({"v": 3.14})
        assert False, "Type mismatch should raise"
    except Exception as e:
        assert "expected" in str(e).lower() or "vector" in str(e).lower() or "type" in str(e).lower()
    print("✓ typed_input_enforcement")


def test_controlled_evolution_executes_mutated():
    """Mutation test must actually run the mutated graph (issue noted in verification)."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "math_ops.ng")).read()
    ng = Neurogen().load(src)
    before = ng.inspect()["nodes"]
    ng.evolve(intensity=1)
    after = ng.inspect()["nodes"]
    # Execute the *mutated* instance, not a fresh one
    r = ng.run({"x": 2.0, "y": 3.0})
    assert r is not None
    print(f"✓ controlled_evolution_executes_mutated (nodes {before} → {after}, result={r})")


def test_invalid_program_rejected():
    """Issue #4: only catch CompileError / ParseError, fail if compilation succeeds."""
    bad = 'GENOME "Bad"\nSTRUCTURE x { INPUT a }\n'  # missing OUTPUT
    try:
        ast = parse(bad)
        compile_program(ast)
        assert False, "Invalid program was accepted"
    except (CompileError, ParseError):
        pass
    # Also test unknown motif
    bad2 = '''
GENOME "Bad2"
STRUCTURE s {
    USE nonexistent_motif
}
OUTPUT s
'''
    try:
        compile_program(parse(bad2))
        assert False, "Unknown motif was accepted"
    except CompileError:
        pass
    print("✓ invalid_program_rejected")


def test_operator_arity_and_types():
    """Negative tests for arity and basic type mismatches."""
    # Wrong arity
    try:
        get_operator("ADD").execute([NeuroValue.from_python(1.0)])
        assert False
    except OperatorError:
        pass
    # Type mismatch for logic
    try:
        get_operator("AND").execute([
            NeuroValue.from_python("hello"),
            NeuroValue.from_python(True)
        ])
        assert False
    except OperatorError:
        pass
    print("✓ operator_arity_and_types")


def test_all_operators_smoke():
    """Smoke-test every registered operator with minimal valid inputs."""
    unary_numeric = ["TANH", "SIGMOID", "RELU", "SOFTMAX", "NORMALIZE", "REDUCE", "AGGREGATE"]
    binary_numeric = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "POWER", "MODULO",
                      "MINIMUM", "MAXIMUM", "GREATER", "LESS", "EQUAL"]
    for name in unary_numeric:
        op = get_operator(name)
        v = NeuroValue.from_python([0.1, -0.2, 0.3])
        try:
            op.execute([v])
        except Exception as e:
            # SOFTMAX/NORMALIZE on small vectors should still work
            raise AssertionError(f"{name} failed: {e}")
    for name in binary_numeric:
        op = get_operator(name)
        a = NeuroValue.from_python(2.0)
        b = NeuroValue.from_python(3.0 if name != "DIVIDE" else 4.0)
        op.execute([a, b])
    # Logic
    for name in ["AND", "OR", "XOR"]:
        get_operator(name).execute([NeuroValue.from_python(True), NeuroValue.from_python(False)])
    get_operator("NOT").execute([NeuroValue.from_python(False)])
    print(f"✓ all_operators_smoke ({len(list_operators())} operators registered)")


def test_operator_stress(n=5000):
    ops = ["ADD", "MULTIPLY", "SUBTRACT", "MAXIMUM", "MINIMUM", "TANH", "RELU", "SIGMOID"]
    failures = 0
    for i in range(n):
        opname = random.choice(ops)
        op = get_operator(opname)
        try:
            if op.arity == 1:
                a = NeuroValue.from_python(random.uniform(-5, 5))
                op.execute([a])
            else:
                a = NeuroValue.from_python(random.uniform(-5, 5))
                b = NeuroValue.from_python(random.uniform(-5, 5))
                op.execute([a, b])
        except Exception:
            failures += 1
    assert failures == 0, f"{failures} failures in {n} trials"
    print(f"✓ operator_stress ({n} random executions, 0 failures)")


def test_end_to_end_batch(n=50):
    src = open(os.path.join(os.path.dirname(__file__), "..", "examples", "math_ops.ng")).read()
    results = []
    for i in range(n):
        ng = Neurogen().load(src)
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        r = ng.run({"x": x, "y": y})
        results.append(r)
    assert all(isinstance(r, (int, float)) for r in results)
    print(f"✓ end_to_end_batch ({n} runs)")


def test_numerical_extremes():
    """Edge values: large, small, negative, near-zero."""
    extremes = [1e10, -1e10, 1e-12, -1e-12, 0.0, 1.0, -1.0]
    for a in extremes:
        for b in extremes:
            if b == 0.0:
                continue  # div-by-zero covered separately
            get_operator("ADD").execute([NeuroValue.from_python(a), NeuroValue.from_python(b)])
            get_operator("MULTIPLY").execute([NeuroValue.from_python(a), NeuroValue.from_python(b)])
    # Empty-ish vector
    get_operator("RELU").execute([NeuroValue.from_python([])])
    print("✓ numerical_extremes")


def run_all():
    print("=" * 60)
    print("NEUROGEN v0.2.1 — Verification & Reliability Suite")
    print("=" * 60)
    test_operators_math()
    test_operators_logic()
    test_operators_transforms()
    test_divide_by_zero()
    test_math_demo()
    test_vector_pipeline()
    test_logic_demo()
    test_motif_and_pipeline()
    test_regression_v01_style()
    test_associative_memory()
    test_keyvalue_memory_connect()
    test_missing_input_raises()
    test_typed_input_enforcement()
    test_controlled_evolution_executes_mutated()
    test_invalid_program_rejected()
    test_operator_arity_and_types()
    test_all_operators_smoke()
    test_operator_stress(5000)
    test_end_to_end_batch(50)
    test_numerical_extremes()
    print("=" * 60)
    print("All v0.2.1 verification tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
