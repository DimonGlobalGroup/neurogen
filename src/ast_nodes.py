"""
Abstract Syntax Tree nodes for NEUROGEN v0.2
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple


@dataclass
class Node:
    """Base AST node."""
    pass


@dataclass
class Genome(Node):
    name: str
    statements: List[Node] = field(default_factory=list)


@dataclass
class MemoryDecl(Node):
    kind: str  # "associative" | "keyvalue"


@dataclass
class Structure(Node):
    name: str
    body: List[Node] = field(default_factory=list)


@dataclass
class Motif(Node):
    name: str
    body: List[Node] = field(default_factory=list)


@dataclass
class InputStmt(Node):
    identifier: str
    type_name: Optional[str] = None  # FLOAT, VECTOR, etc.


@dataclass
class ConstStmt(Node):
    identifier: str
    value: Any
    type_name: Optional[str] = None


@dataclass
class LetStmt(Node):
    """LET name = OPERATOR(arg1, arg2, ...) or LET name = identifier"""
    identifier: str
    operator: Optional[str] = None
    args: List[Union[str, float, int, bool]] = field(default_factory=list)


@dataclass
class TransformStmt(Node):
    identifier: str


@dataclass
class ConnectStmt(Node):
    identifier: str


@dataclass
class EvaluateStmt(Node):
    identifier: str


@dataclass
class ObserveStmt(Node):
    identifier: str


@dataclass
class AdaptStmt(Node):
    target: str


@dataclass
class UseStmt(Node):
    motif_name: str


@dataclass
class OutputStmt(Node):
    identifier: str


@dataclass
class StructureOutput(Node):
    """OUTPUT <name> inside a STRUCTURE body: declares a named structure output
    bound to a previously bound identifier (LET/CONST/INPUT name)."""
    name: str


@dataclass
class PipelineConnect(Node):
    """CONNECT <struct>.<output> TO <struct>.<input> inside a PIPELINE block."""
    src_struct: str
    src_output: str
    dst_struct: str
    dst_input: str


@dataclass
class Pipeline(Node):
    """Top-level PIPELINE declaration."""
    name: str
    connects: List[PipelineConnect] = field(default_factory=list)
    outputs: List[Tuple[str, str]] = field(default_factory=list)  # (struct, output)


@dataclass
class CallStmt(Node):
    """Direct operator call: ADD(x, y)"""
    operator: str
    args: List[Union[str, float, int, bool]] = field(default_factory=list)
    result_name: Optional[str] = None


# ---- Intermediate Representation ----

@dataclass
class GraphNode:
    """A node in the rooted computational graph."""
    id: str
    op: str
    args: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputationalGraph:
    """Rooted typed computational graph produced by the compiler."""
    genome_name: str
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)
    output_id: Optional[str] = None
    memory_kind: str = "associative"
    motifs: Dict[str, Any] = field(default_factory=dict)
    structure_nodes: Dict[str, List[str]] = field(default_factory=dict)
    structure_outputs: Dict[str, str] = field(default_factory=dict)
    # v0.3: per-structure declared OUTPUT name -> node id, for independently
    # compiled structures. {structure_name: {output_name: node_id}}
    structure_declared_outputs: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def topological_order(self) -> List[str]:
        from collections import defaultdict, deque
        indeg = defaultdict(int)
        children = defaultdict(list)
        for a, b in self.edges:
            children[a].append(b)
            indeg[b] += 1
        for nid in self.nodes:
            indeg.setdefault(nid, 0)

        q = deque([n for n in self.nodes if indeg[n] == 0])
        order = []
        while q:
            n = q.popleft()
            order.append(n)
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in computational graph")
        return order


# ---- v0.3 Pipeline intermediate representation ----

@dataclass
class Connection:
    """A validated, compiled connection between two structures."""
    src_struct: str
    src_output: str
    dst_struct: str
    dst_input: str
    src_dtype: Any = None   # DType of the source output (inferred)
    dst_dtype: Any = None   # DType expected by the destination input


@dataclass
class PipelineIR:
    """Compiled pipeline: everything the engine needs to execute without
    re-interpreting source."""
    name: str
    structures: Dict[str, Any] = field(default_factory=dict)  # name -> CompiledStructure
    connections: List[Connection] = field(default_factory=list)
    dep_edges: List[Tuple[str, str]] = field(default_factory=list)
    exec_order: List[str] = field(default_factory=list)
    outputs: List[Tuple[str, str]] = field(default_factory=list)  # (struct, output)
    meta: Dict[str, Any] = field(default_factory=dict)
