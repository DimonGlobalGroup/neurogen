"""
NEUROGEN v0.3 — Pipeline compiler.

Validates PIPELINE declarations and produces a PipelineIR: structures,
validated connections, dependency edges, execution order, and outputs.
All static validation happens here, before any runtime execution.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set

from ast_nodes import Genome, Pipeline, PipelineConnect, PipelineIR, Connection, Structure
from compiler import CompileError
from structure import CompiledStructure, compile_structure
from neuro_types import DType
from errors import (
    NeurogenError, StructureError, ConnectionError, PipelineError,
)


def types_compatible(src: DType, dst: DType) -> bool:
    """Compile-time connection type rule (v0.3).

    Exact dtype match required. The only promotion allowed is INT -> FLOAT,
    consistent with the v0.2.1 input rule. ANY on either side defers the
    check to runtime validation (defense in depth in the engine).
    """
    if src == dst:
        return True
    if src == DType.ANY or dst == DType.ANY:
        return True
    if src == DType.INT and dst == DType.FLOAT:
        return True
    return False


@dataclass
class PipelineLimits:
    max_structures: int = 64
    max_connections: int = 1024
    max_depth: int = 32


class PipelineCompiler:
    def __init__(self, genome: Genome,
                 structures: Optional[Dict[str, CompiledStructure]] = None,
                 limits: Optional[PipelineLimits] = None):
        self.genome = genome
        self.structures: Dict[str, CompiledStructure] = dict(structures or {})
        self.limits = limits or PipelineLimits()
        self._known = {s.name for s in genome.statements
                       if isinstance(s, Structure)}

    def _require_structure(self, name: str, where: str) -> CompiledStructure:
        if name not in self._known:
            raise ConnectionError(
                f"{where}: unknown STRUCTURE '{name}'")
        if name not in self.structures:
            self.structures[name] = compile_structure(self.genome, name)
        return self.structures[name]

    def compile(self, pipeline: Pipeline) -> PipelineIR:
        connections: List[Connection] = []
        seen_dst: Dict[Tuple[str, str], PipelineConnect] = {}
        dep_edges: Set[Tuple[str, str]] = set()
        participants: Set[str] = set()

        for pc in pipeline.connects:
            src_cs = self._require_structure(
                pc.src_struct, f"PIPELINE '{pipeline.name}' CONNECT")
            dst_cs = self._require_structure(
                pc.dst_struct, f"PIPELINE '{pipeline.name}' CONNECT")

            if pc.src_output not in src_cs.outputs:
                raise ConnectionError(
                    f"PIPELINE '{pipeline.name}': STRUCTURE '{pc.src_struct}' "
                    f"has no OUTPUT '{pc.src_output}' "
                    f"(declared: {sorted(src_cs.outputs)})")
            dst_input_names = [n for n, _ in dst_cs.inputs]
            if pc.dst_input not in dst_input_names:
                raise ConnectionError(
                    f"PIPELINE '{pipeline.name}': STRUCTURE '{pc.dst_struct}' "
                    f"has no INPUT '{pc.dst_input}' "
                    f"(declared: {dst_input_names})")

            src_dt = src_cs.output_dtypes[pc.src_output]
            dst_dt = dst_cs.input_dtype(pc.dst_input)
            if not types_compatible(src_dt, dst_dt):
                raise ConnectionError(
                    f"PIPELINE '{pipeline.name}': type mismatch: "
                    f"{pc.src_struct}.{pc.src_output} ({src_dt.name}) -> "
                    f"{pc.dst_struct}.{pc.dst_input} ({dst_dt.name}); "
                    f"only identical types or INT->FLOAT are allowed")

            key = (pc.dst_struct, pc.dst_input)
            if key in seen_dst:
                prev = seen_dst[key]
                raise ConnectionError(
                    f"PIPELINE '{pipeline.name}': duplicate connection to "
                    f"{pc.dst_struct}.{pc.dst_input} "
                    f"(already connected from {prev.src_struct}.{prev.src_output})")
            seen_dst[key] = pc

            connections.append(Connection(
                src_struct=pc.src_struct, src_output=pc.src_output,
                dst_struct=pc.dst_struct, dst_input=pc.dst_input,
                src_dtype=src_dt, dst_dtype=dst_dt))
            dep_edges.add((pc.src_struct, pc.dst_struct))
            participants.add(pc.src_struct)
            participants.add(pc.dst_struct)

        if len(participants) > self.limits.max_structures:
            raise PipelineError(
                f"PIPELINE '{pipeline.name}': {len(participants)} structures "
                f"exceeds limit of {self.limits.max_structures}")
        if len(connections) > self.limits.max_connections:
            raise PipelineError(
                f"PIPELINE '{pipeline.name}': {len(connections)} connections "
                f"exceeds limit of {self.limits.max_connections}")

        outputs: List[Tuple[str, str]] = []
        for sname, oname in pipeline.outputs:
            cs = self._require_structure(
                sname, f"PIPELINE '{pipeline.name}' OUTPUT")
            if oname not in cs.outputs:
                raise ConnectionError(
                    f"PIPELINE '{pipeline.name}': STRUCTURE '{sname}' "
                    f"has no OUTPUT '{oname}' "
                    f"(declared: {sorted(cs.outputs)})")
            outputs.append((sname, oname))
            participants.add(sname)

        exec_order = self._topological_order(pipeline.name, participants, dep_edges)
        depth = self._longest_path(exec_order, dep_edges)
        if depth > self.limits.max_depth:
            raise PipelineError(
                f"PIPELINE '{pipeline.name}': dependency depth {depth} "
                f"exceeds limit of {self.limits.max_depth}")
        self._check_reachable(pipeline.name, participants, dep_edges, outputs)

        return PipelineIR(
            name=pipeline.name,
            structures={n: self.structures[n] for n in participants},
            connections=connections,
            dep_edges=sorted(dep_edges),
            exec_order=exec_order,
            outputs=outputs,
            meta={
                "num_structures": len(participants),
                "num_connections": len(connections),
                "depth": depth,
                "genome": self.genome.name,
            },
        )

    def compile_all(self) -> Dict[str, PipelineIR]:
        result = {}
        for stmt in self.genome.statements:
            if isinstance(stmt, Pipeline):
                if stmt.name in result:
                    raise PipelineError(f"Duplicate PIPELINE '{stmt.name}'")
                result[stmt.name] = self.compile(stmt)
        return result

    # ---- internal helpers ----

    def _topological_order(self, pname: str, participants: Set[str],
                           dep_edges: Set[Tuple[str, str]]) -> List[str]:
        children: Dict[str, List[str]] = defaultdict(list)
        indeg: Dict[str, int] = {p: 0 for p in participants}
        for a, b in dep_edges:
            children[a].append(b)
            indeg[b] += 1
        q = deque(sorted(p for p in participants if indeg[p] == 0))
        order: List[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for c in sorted(children[n]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(participants):
            cycle = self._find_cycle(participants, dep_edges)
            raise PipelineError(
                f"PIPELINE '{pname}': circular dependency detected: "
                f"{' -> '.join(cycle)}; v0.3 rejects cycles with diagnostics "
                f"instead of uncontrolled iterative execution")
        return order

    def _find_cycle(self, participants: Set[str],
                    dep_edges: Set[Tuple[str, str]]) -> List[str]:
        children: Dict[str, List[str]] = defaultdict(list)
        for a, b in dep_edges:
            children[a].append(b)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {p: WHITE for p in participants}
        stack: List[str] = []

        def dfs(n: str) -> Optional[List[str]]:
            color[n] = GRAY
            stack.append(n)
            for c in sorted(children[n]):
                if color[c] == GRAY:
                    return stack[stack.index(c):] + [c]
                if color[c] == WHITE:
                    hit = dfs(c)
                    if hit:
                        return hit
            stack.pop()
            color[n] = BLACK
            return None

        for p in sorted(participants):
            if color[p] == WHITE:
                hit = dfs(p)
                if hit:
                    return hit
        return sorted(participants)

    def _longest_path(self, order: List[str],
                      dep_edges: Set[Tuple[str, str]]) -> int:
        dist = {n: 0 for n in order}
        children: Dict[str, List[str]] = defaultdict(list)
        for a, b in dep_edges:
            children[a].append(b)
        for n in order:
            for c in children[n]:
                if dist[c] < dist[n] + 1:
                    dist[c] = dist[n] + 1
        return max(dist.values(), default=0)

    def _check_reachable(self, pname: str, participants: Set[str],
                         dep_edges: Set[Tuple[str, str]],
                         outputs: List[Tuple[str, str]]) -> None:
        # Every participating structure must lie on a path to a declared
        # pipeline output; otherwise it can never affect the result and the
        # declaration is almost certainly a mistake.
        parents: Dict[str, List[str]] = defaultdict(list)
        for a, b in dep_edges:
            parents[b].append(a)
        sinks = {s for s, _ in outputs}
        seen: Set[str] = set()
        q = deque(sorted(sinks))
        while q:
            n = q.popleft()
            if n in seen:
                continue
            seen.add(n)
            for p in parents[n]:
                q.append(p)
        unreachable = sorted(participants - seen)
        if unreachable:
            raise PipelineError(
                f"PIPELINE '{pname}': unreachable STRUCTUREs "
                f"{unreachable}: every participating structure must lie on a "
                f"path to a declared pipeline OUTPUT")
