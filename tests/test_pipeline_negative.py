"""
NEUROGEN v0.3 — Negative test suite for cross-structure pipelines.

Every test asserts the CORRECT error class / behavior, not merely that
"an exception occurred".
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse, ParseError
from compiler import CompileError
from pipeline import PipelineCompiler, PipelineLimits
from pipeline_engine import PipelineEngine, EngineSettings
from interface import NeurogenPipeline
from ast_nodes import Pipeline
from errors import (StructureError, ConnectionError, PipelineError,
                    StructureInputError, StructureExecutionError)


BASE = '''
GENOME "N"
STRUCTURE a {
    INPUT x : FLOAT
    LET y = MULTIPLY(x, 2.0)
    OUTPUT y
}
STRUCTURE b {
    INPUT z : FLOAT
    LET r = ADD(z, 10.0)
    OUTPUT r
}
'''


def _compile(src):
    g = parse(src)
    p = [s for s in g.statements if isinstance(s, Pipeline)][0]
    return PipelineCompiler(g), p


def test_unknown_structure_in_connect():
    pc, p = _compile(BASE + 'PIPELINE m { CONNECT a.y TO ghost.z\n OUTPUT b.r\n}\n')
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "ghost" in str(e)


def test_unknown_structure_in_output():
    pc, p = _compile(BASE + 'PIPELINE m { CONNECT a.y TO b.z\n OUTPUT ghost.r\n}\n')
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "ghost" in str(e)


def test_unknown_pipeline_run():
    ngp = NeurogenPipeline().load(BASE + 'PIPELINE m { OUTPUT a.y\n}\n')
    try:
        ngp.run("nope", {})
        raise AssertionError("accepted")
    except KeyError as e:
        assert "nope" in str(e)


def test_unknown_input():
    pc, p = _compile(BASE + 'PIPELINE m { CONNECT a.y TO b.nope\n OUTPUT b.r\n}\n')
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "nope" in str(e) and "INPUT" in str(e)


def test_unknown_output():
    pc, p = _compile(BASE + 'PIPELINE m { CONNECT a.nope TO b.z\n OUTPUT b.r\n}\n')
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "nope" in str(e) and "OUTPUT" in str(e)


def test_missing_input_runtime():
    ngp = NeurogenPipeline().load(BASE + 'PIPELINE m { OUTPUT a.y\n}\n')
    r = ngp.run("m", {})  # a.x not provided
    assert r.status == "failed"
    assert r.structure_status["a"] == "failed"
    err = r.trace[0].error
    assert err["class"] == "StructureInputError", err
    assert "x" in err["message"]


def test_invalid_type_vector_to_float():
    src = BASE + '''
STRUCTURE v {
    INPUT t : VECTOR
    LET n = NORMALIZE(t)
    OUTPUT n
}
PIPELINE m { CONNECT v.n TO b.z
 OUTPUT b.r
}
'''
    pc, p = _compile(src)
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "VECTOR" in str(e) and "FLOAT" in str(e)


def test_invalid_type_float_to_int():
    src = BASE + '''
STRUCTURE i {
    INPUT k : INT
    OUTPUT k
}
PIPELINE m { CONNECT a.y TO i.k
 OUTPUT i.k
}
'''
    pc, p = _compile(src)
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "FLOAT" in str(e) and "INT" in str(e)


def test_invalid_type_bool_to_float():
    src = BASE + '''
STRUCTURE lb {
    INPUT q : BOOL
    OUTPUT q
}
PIPELINE m { CONNECT lb.q TO b.z
 OUTPUT b.r
}
'''
    pc, p = _compile(src)
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "BOOL" in str(e)


def test_invalid_connection_string_literal():
    # CONNECT with a malformed (non-dotted) reference is a parse error
    try:
        parse('GENOME "X"\nSTRUCTURE a { INPUT x : FLOAT\n OUTPUT x }\n'
              'PIPELINE m { CONNECT a TO b.z\n OUTPUT a.x\n}\n')
        raise AssertionError("accepted")
    except ParseError:
        pass


def test_duplicate_connection():
    pc, p = _compile(BASE + 'PIPELINE m { CONNECT a.y TO b.z\n CONNECT a.y TO b.z\n OUTPUT b.r\n}\n')
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except ConnectionError as e:
        assert "duplicate" in str(e).lower()


def test_circular_two_cycle():
    pc, p = _compile(BASE + 'PIPELINE m { CONNECT a.y TO b.z\n CONNECT b.r TO a.x\n OUTPUT b.r\n}\n')
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "circular" in str(e).lower() and "a -> b -> a" in str(e)


def test_circular_three_cycle():
    src = BASE + '''
STRUCTURE c {
    INPUT q : FLOAT
    LET s = ADD(q, 1.0)
    OUTPUT s
}
PIPELINE m {
    CONNECT a.y TO b.z
    CONNECT b.r TO c.q
    CONNECT c.s TO a.x
    OUTPUT c.s
}
'''
    pc, p = _compile(src)
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "circular" in str(e).lower()


def test_circular_self_loop():
    pc, p = _compile(BASE + 'PIPELINE m { CONNECT a.y TO a.x\n OUTPUT a.y\n}\n')
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "circular" in str(e).lower()


def test_invalid_pipeline_syntax_missing_to():
    try:
        parse('GENOME "X"\nSTRUCTURE a { INPUT x : FLOAT\n OUTPUT x }\n'
              'PIPELINE m { CONNECT a.x b.x\n OUTPUT a.x\n}\n')
        raise AssertionError("accepted")
    except ParseError:
        pass


def test_pipeline_without_output():
    try:
        parse('GENOME "X"\nSTRUCTURE a { INPUT x : FLOAT\n OUTPUT x }\n'
              'PIPELINE m { CONNECT a.x TO a.x\n}\n')
        raise AssertionError("accepted")
    except ParseError:
        pass


def test_unreachable_structure():
    src = BASE + '''
STRUCTURE lone {
    INPUT x : FLOAT
    LET y = ADD(x, 1.0)
    OUTPUT y
}
PIPELINE m {
    CONNECT a.y TO b.z
    CONNECT a.y TO lone.x
    OUTPUT b.r
}
'''
    pc, p = _compile(src)
    try:
        pc.compile(p)
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "unreachable" in str(e).lower() and "lone" in str(e)


def test_conflicting_output_definitions():
    # duplicate OUTPUT name inside one structure
    try:
        parse('GENOME "X"\nSTRUCTURE a { INPUT x : FLOAT\n OUTPUT x\n OUTPUT x\n}\n')
        from structure import compile_structure
        g = parse('GENOME "X"\nSTRUCTURE a { INPUT x : FLOAT\n OUTPUT x\n OUTPUT x\n}\n')
        compile_structure(g, "a")
        raise AssertionError("accepted")
    except CompileError as e:
        assert "uplicate" in str(e)


def test_duplicate_pipeline_name():
    src = BASE + 'PIPELINE m { OUTPUT a.y\n}\nPIPELINE m { OUTPUT b.r\n}\n'
    g = parse(src)
    try:
        PipelineCompiler(g).compile_all()
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "uplicate" in str(e)


def test_upstream_failure_blocks():
    ngp = NeurogenPipeline().load(BASE + '''
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
PIPELINE m {
    CONNECT bad.r TO down.q
    OUTPUT down.s
}
''')
    r = ngp.run("m", {"bad.x": 1.0})
    assert r.status == "failed"
    assert r.structure_status == {"bad": "failed", "down": "blocked"}, r.structure_status
    assert r.outputs == {}


def test_downstream_failure_at_sink():
    ngp = NeurogenPipeline().load(BASE + '''
STRUCTURE bad {
    INPUT x : FLOAT
    CONST zero = 0.0
    LET r = DIVIDE(x, zero)
    OUTPUT r
}
PIPELINE m {
    CONNECT a.y TO bad.x
    OUTPUT bad.r
}
''')
    r = ngp.run("m", {"a.x": 2.0})
    assert r.status == "failed"
    assert r.structure_status["a"] == "success"
    assert r.structure_status["bad"] == "failed"
    err = [t.error for t in r.trace if t.name == "bad"][0]
    assert err["class"] == "StructureExecutionError"


def test_malformed_source():
    for bad in ['this is not neurogen {{{', 'STRUCTURE {', 'PIPELINE 123 { }']:
        try:
            parse(bad)
            raise AssertionError(f"accepted: {bad!r}")
        except (ParseError, Exception) as e:
            assert isinstance(e, ParseError), (bad, type(e).__name__)


def test_oversized_pipeline():
    g = parse(BASE + 'PIPELINE m { CONNECT a.y TO b.z\n OUTPUT b.r\n}\n')
    p = [s for s in g.statements if isinstance(s, Pipeline)][0]
    try:
        PipelineCompiler(g, limits=PipelineLimits(max_structures=1)).compile(p)
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "exceeds limit" in str(e)


def test_oversized_connections():
    g = parse(BASE + 'PIPELINE m { CONNECT a.y TO b.z\n OUTPUT b.r\n}\n')
    p = [s for s in g.statements if isinstance(s, Pipeline)][0]
    try:
        PipelineCompiler(g, limits=PipelineLimits(max_connections=0)).compile(p)
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "exceeds limit" in str(e)


def test_invalid_tensor_dims_runtime():
    """MATRIX-declared input receiving a VECTOR must fail at runtime."""
    src = '''
GENOME "TD"
STRUCTURE m {
    INPUT t : MATRIX
    LET s = REDUCE(t)
    OUTPUT s
}
PIPELINE main {
    OUTPUT m.s
}
'''
    ngp = NeurogenPipeline().load(src)
    r = ngp.run("main", {"m.t": [1.0, 2.0, 3.0]})  # VECTOR, not MATRIX
    assert r.status == "failed"
    assert r.structure_status["m"] == "failed"
    err = r.trace[0].error
    assert err["class"] in ("StructureInputError", "StructureExecutionError"), err


def test_node_budget_violation():
    """max_steps smaller than the graph -> deterministic budget failure."""
    ngp = NeurogenPipeline().load(BASE + 'PIPELINE m { OUTPUT a.y\n}\n')
    r = ngp.run("m", {"a.x": 1.0}, settings=EngineSettings(max_steps=1))
    assert r.status == "failed"
    assert r.structure_status["a"] == "failed"
    err = r.trace[0].error
    assert err["class"] == "StructureExecutionError", err
    assert "too large" in err["message"].lower()


def test_total_budget_violation():
    ngp = NeurogenPipeline().load(BASE + 'PIPELINE m { OUTPUT a.y\n}\n')
    try:
        ngp.run("m", {"a.x": 1.0}, settings=EngineSettings(total_budget_s=0.0))
        raise AssertionError("accepted")
    except PipelineError as e:
        assert "budget" in str(e).lower()


def test_output_of_unknown_identifier():
    try:
        parse('GENOME "X"\nSTRUCTURE a { INPUT x : FLOAT\n OUTPUT ghost\n}\n')
        from structure import compile_structure
        g = parse('GENOME "X"\nSTRUCTURE a { INPUT x : FLOAT\n OUTPUT ghost\n}\n')
        compile_structure(g, "a")
        raise AssertionError("accepted")
    except CompileError as e:
        assert "ghost" in str(e)


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
    print(f"\nnegative: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
