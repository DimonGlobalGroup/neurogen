"""
NEUROGEN v0.3 — Structure interface and independent compilation.

A structure is an independently compilable and executable computational unit
with a declared interface: typed inputs and named, typed outputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import hashlib

from ast_nodes import (
    Node, Genome, Structure, Motif, InputStmt, ComputationalGraph,
)
from compiler import compile_structure_body, CompileError
from neuro_types import NeuroValue, DType, promote_type
from operators import REGISTRY


def dtype_from_name(name: Optional[str]) -> DType:
    if not name:
        return DType.ANY
    return {
        "BOOL": DType.BOOL, "INT": DType.INT, "FLOAT": DType.FLOAT,
        "STRING": DType.STRING, "VECTOR": DType.VECTOR,
        "MATRIX": DType.MATRIX, "TENSOR": DType.TENSOR,
        "OBJECT": DType.OBJECT, "ANY": DType.ANY,
    }.get(name.upper(), DType.ANY)


_ELEMENTWISE_BINARY = {
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE",
    "POWER", "MODULO", "MINIMUM", "MAXIMUM",
}
_UNARY_PRESERVE = {"TANH", "SIGMOID", "RELU"}


def _infer_op_dtype(opname: str, arg_dtypes: List[DType]) -> DType:
    """Best-effort static dtype inference for one operator application.

    This is an approximation used for compile-time connection validation.
    The pipeline engine always re-validates actual runtime values, so an
    imprecise inference degrades to a runtime check, never to silent
    acceptance of a wrong value.
    """
    if opname in _ELEMENTWISE_BINARY:
        result = DType.INT
        for t in arg_dtypes:
            result = promote_type(result, t)
        return result
    if opname in _UNARY_PRESERVE or opname in ("NORMALIZE",):
        return arg_dtypes[0] if arg_dtypes else DType.ANY
    if opname == "CLAMP":
        return arg_dtypes[0] if arg_dtypes else DType.ANY
    if opname == "SOFTMAX":
        return DType.VECTOR
    if opname in ("AND", "OR", "NOT", "XOR", "EQUAL", "GREATER", "LESS"):
        if all(t in (DType.BOOL, DType.INT, DType.FLOAT) for t in arg_dtypes):
            return DType.BOOL
        return DType.ANY  # elementwise over arrays; defer to runtime
    if opname in ("REDUCE", "AGGREGATE"):
        return DType.FLOAT
    if opname == "VECTOR":
        return DType.VECTOR
    if opname == "MATRIX":
        return DType.MATRIX
    if opname == "INDEX":
        return DType.ANY
    if opname in ("SLICE", "CONCATENATE", "MAP"):
        return DType.VECTOR
    if opname == "MERGE":
        return DType.OBJECT
    if opname in ("TRANSFORM", "ROUTE"):
        return DType.ANY
    if opname in ("CONNECT", "EVALUATE", "OBSERVE", "ADAPT"):
        return DType.OBJECT
    if opname == "IDENTITY":
        return arg_dtypes[0] if arg_dtypes else DType.ANY
    if opname in REGISTRY:
        return REGISTRY[opname].output_type
    return DType.ANY


def infer_node_dtype(graph: ComputationalGraph, node_id: str,
                     _memo: Optional[Dict[str, DType]] = None) -> DType:
    """Infer the dtype produced by a graph node (best effort, memoized)."""
    memo = _memo if _memo is not None else {}
    if node_id in memo:
        return memo[node_id]
    node = graph.nodes[node_id]
    op = node.op
    if op == "input":
        result = dtype_from_name(node.meta.get("type"))
    elif op == "const":
        result = NeuroValue.from_python(node.args[0]).dtype
    elif op == "output":
        preds = [a for a, b in graph.edges if b == node.id]
        result = infer_node_dtype(graph, preds[0], memo) if preds else DType.ANY
    else:
        arg_dtypes: List[DType] = []
        for a in node.args:
            if isinstance(a, tuple) and len(a) == 2:
                kind, val = a
                if kind == "ref" and val in graph.nodes:
                    arg_dtypes.append(infer_node_dtype(graph, val, memo))
                else:
                    arg_dtypes.append(NeuroValue.from_python(val).dtype)
            else:
                arg_dtypes.append(NeuroValue.from_python(a).dtype)
        if not arg_dtypes:
            # compiler may wire predecessors implicitly (all-literal calls)
            preds = [a for a, b in graph.edges if b == node.id]
            for p in preds:
                arg_dtypes.append(infer_node_dtype(graph, p, memo))
        result = _infer_op_dtype(op, arg_dtypes)
    memo[node_id] = result
    return result


def graph_definition_hash(graph: ComputationalGraph) -> str:
    """Canonical hash of a compiled structure graph (for cache keys)."""
    parts = []
    for nid in sorted(graph.nodes):
        n = graph.nodes[nid]
        parts.append(f"{nid}|{n.op}|{n.args!r}")
    for a, b in sorted(graph.edges):
        parts.append(f"{a}->{b}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


@dataclass
class CompiledStructure:
    """An independently compiled structure with its declared interface."""
    name: str
    graph: ComputationalGraph
    inputs: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    input_node_ids: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)       # output name -> node id
    output_dtypes: Dict[str, DType] = field(default_factory=dict)
    def_hash: str = ""
    has_memory_write: bool = False                              # contains CONNECT
    memory_kind: str = "associative"

    def input_dtype(self, name: str) -> DType:
        for iname, tname in self.inputs:
            if iname == name:
                return dtype_from_name(tname)
        raise KeyError(name)


def reinterface(name: str, graph: ComputationalGraph,
                inputs: List[Tuple[str, Optional[str]]],
                outputs: Dict[str, str],
                input_node_ids: Optional[Dict[str, str]] = None,
                ) -> CompiledStructure:
    """Rebuild interface metadata after external graph surgery (e.g. the
    v0.2.1 ControlledEvolver mutating the graph in place).

    Node ids that no longer exist are dropped; dtypes, definition hash, and
    the memory-write flag are recomputed from the mutated graph.
    """
    outputs = {o: nid for o, nid in outputs.items() if nid in graph.nodes}
    keep_ids = {n: nid for n, nid in (input_node_ids or {}).items()
                if nid in graph.nodes}
    return CompiledStructure(
        name=name,
        graph=graph,
        inputs=list(inputs),
        input_node_ids=keep_ids,
        outputs=outputs,
        output_dtypes={o: infer_node_dtype(graph, nid)
                       for o, nid in outputs.items()},
        def_hash=graph_definition_hash(graph),
        has_memory_write=any(n.op == "CONNECT" for n in graph.nodes.values()),
        memory_kind=graph.memory_kind,
    )


def compile_structure(genome: Genome, name: str) -> CompiledStructure:
    """Compile one STRUCTURE from a parsed genome into an independent unit."""
    struct = None
    motifs: Dict[str, Motif] = {}
    memory_kind = "associative"
    for stmt in genome.statements:
        if isinstance(stmt, Structure) and stmt.name == name:
            struct = stmt
        elif isinstance(stmt, Motif):
            motifs[stmt.name] = stmt
        elif type(stmt).__name__ == "MemoryDecl":
            memory_kind = stmt.kind
    if struct is None:
        raise CompileError(f"Unknown STRUCTURE '{name}'")

    graph, local_map, declared = compile_structure_body(
        struct, motifs, genome.name, memory_kind=memory_kind)

    inputs: List[Tuple[str, Optional[str]]] = []
    input_node_ids: Dict[str, str] = {}
    for stmt in struct.body:
        if isinstance(stmt, InputStmt):
            inputs.append((stmt.identifier, stmt.type_name))
            if stmt.identifier in local_map:
                input_node_ids[stmt.identifier] = local_map[stmt.identifier]

    output_dtypes = {oname: infer_node_dtype(graph, nid)
                     for oname, nid in declared.items()}
    has_write = any(n.op == "CONNECT" for n in graph.nodes.values())

    return CompiledStructure(
        name=name,
        graph=graph,
        inputs=inputs,
        input_node_ids=input_node_ids,
        outputs=dict(declared),
        output_dtypes=output_dtypes,
        def_hash=graph_definition_hash(graph),
        has_memory_write=has_write,
        memory_kind=memory_kind,
    )
