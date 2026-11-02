"""
NEUROGEN v0.3 — Mutation compatibility tests.

The v0.2.1 ControlledEvolver must keep working on independently compiled
structure graphs, and pipeline connection validation must keep working
against (possibly mutated) structure interfaces.
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse
from structure import compile_structure, reinterface, infer_node_dtype
from pipeline import PipelineCompiler
from pipeline_engine import PipelineEngine, EngineSettings
from interface import NeurogenPipeline
from evolution import ControlledEvolver
from runtime import Runtime
from ast_nodes import Pipeline
from neuro_types import DType
from errors import ConnectionError


def _genome():
    return parse('''
GENOME "MG"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    LET z = MULTIPLY(y, 2.0)
    OUTPUT z
}
STRUCTURE b {
    INPUT q : FLOAT
    LET r = ADD(q, 5.0)
    OUTPUT r
}
PIPELINE main {
    CONNECT a.z TO b.q
    OUTPUT b.r
}
''')


def test_mutation_produces_valid_executable_graph():
    g = _genome()
    cs = compile_structure(g, "a")
    for seed in (1, 2, 3):
        random.seed(seed)
        evo = ControlledEvolver()
        mut = evo.mutate(cs.graph, intensity=2)
        # structurally valid: topological order exists
        order = mut.topological_order()
        assert len(order) == len(mut.nodes)
        # still executable via the standard Runtime
        rt = Runtime(mut)
        res = rt.execute({"x": 3.0})
        assert res is not None
    print("✓ mutation_produces_valid_executable_graph")


def test_mutated_structure_reinterfaces():
    g = _genome()
    cs = compile_structure(g, "a")
    random.seed(7)
    mut = ControlledEvolver().mutate(cs.graph, intensity=2)
    cs2 = reinterface("a", mut, cs.inputs, cs.outputs, cs.input_node_ids)
    assert "z" in cs2.outputs, cs2.outputs
    assert cs2.def_hash != cs.def_hash  # definition actually changed
    # mutated interface still executes standalone
    rt = Runtime(cs2.graph)
    res = rt.execute({"x": 1.0})
    assert res is not None
    print("✓ mutated_structure_reinterfaces")


def test_mutated_structure_in_pipeline():
    """A pipeline built with a mutated structure still validates and runs."""
    g = _genome()
    cs_a = compile_structure(g, "a")
    cs_b = compile_structure(g, "b")
    random.seed(11)
    mut = ControlledEvolver().mutate(cs_a.graph, intensity=1)
    cs_a2 = reinterface("a", mut, cs_a.inputs, cs_a.outputs, cs_a.input_node_ids)
    p = [s for s in g.statements if isinstance(s, Pipeline)][0]
    ir = PipelineCompiler(g, {"a": cs_a2, "b": cs_b}).compile(p)
    eng = PipelineEngine(ir, EngineSettings(cache_enabled=False))
    r = eng.execute({"a.x": 2.0})
    assert r.status == "success", r.errors
    assert r.structure_status["a"] == "success"
    print("✓ mutated_structure_in_pipeline")


def test_interface_rejects_incompatible_mutated_output():
    """Connection validation uses the interface: a mutated/wrong dtype is rejected."""
    g = _genome()
    cs_a = compile_structure(g, "a")
    cs_b = compile_structure(g, "b")
    # simulate a mutation that changed the output dtype: lie in the metadata
    cs_a_bad = reinterface("a", cs_a.graph, cs_a.inputs, cs_a.outputs,
                           cs_a.input_node_ids)
    cs_a_bad.output_dtypes = {"z": DType.VECTOR}
    p = [s for s in g.statements if isinstance(s, Pipeline)][0]
    try:
        PipelineCompiler(g, {"a": cs_a_bad, "b": cs_b}).compile(p)
        raise AssertionError("incompatible mutated interface accepted")
    except ConnectionError as e:
        assert "VECTOR" in str(e) and "FLOAT" in str(e)
    print("✓ interface_rejects_incompatible_mutated_output")


def test_mutation_does_not_corrupt_other_stages():
    """Mutating one structure leaves the rest of the pipeline intact."""
    g = _genome()
    cs_a = compile_structure(g, "a")
    cs_b = compile_structure(g, "b")
    random.seed(21)
    mut = ControlledEvolver().mutate(cs_b.graph, intensity=1)
    cs_b2 = reinterface("b", mut, cs_b.inputs, cs_b.outputs, cs_b.input_node_ids)
    ngp = NeurogenPipeline().load('\n'.join([
        'GENOME "MG"',
        'STRUCTURE a { INPUT x : FLOAT',
        ' LET y = ADD(x, 1.0)',
        ' LET z = MULTIPLY(y, 2.0)',
        ' OUTPUT z }',
        'STRUCTURE b { INPUT q : FLOAT',
        ' LET r = ADD(q, 5.0)',
        ' OUTPUT r }',
        'PIPELINE main { CONNECT a.z TO b.q',
        ' OUTPUT b.r }',
    ]))
    # sanity: unmutated pipeline is correct: ((x+1)*2)+5
    r0 = ngp.run("main", {"a.x": 3.0})
    assert r0.outputs["b.r"] == ((3.0 + 1.0) * 2.0) + 5.0
    # now run with mutated b through a hand-built compiler
    p = [s for s in g.statements if isinstance(s, Pipeline)][0]
    ir = PipelineCompiler(g, {"a": cs_a, "b": cs_b2}).compile(p)
    r = PipelineEngine(ir, EngineSettings(cache_enabled=False)).execute({"a.x": 3.0})
    assert r.status == "success", r.errors
    assert r.structure_status["a"] == "success"
    # structure a (unmutated) still computed its exact value into b
    ta = [t for t in r.trace if t.name == "a"][0]
    assert ta.outputs_summary == {"z": "FLOAT(8.0)"}, ta.outputs_summary
    print("✓ mutation_does_not_corrupt_other_stages")


def run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception:
            print(f"✗ {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\nmutation: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
