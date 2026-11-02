"""
NEUROGEN Compiler v0.2 — builds executable rooted computational graphs
with real data flow between structures and operators.
"""

from typing import Dict, List, Any, Optional, Union
from ast_nodes import *
from operators import REGISTRY


class CompileError(Exception):
    pass




def _resolve_arg(arg: Any, local_map: Dict[str, str]) -> Any:
    """Resolve an argument to either a literal or a node-id reference."""
    if isinstance(arg, (int, float, bool)):
        return ("literal", arg)
    if isinstance(arg, str):
        if arg in local_map:
            return ("ref", local_map[arg])
        # try as operator name or keep as string literal
        return ("literal", arg)
    return ("literal", arg)


def _compile_body(graph: ComputationalGraph, next_id, motifs: Dict[str, Motif],
                    body: List[Node], parent_name: str,
                    incoming: Optional[List[str]] = None,
                    outer_map: Optional[Dict[str, str]] = None,
                    consume_incoming: bool = False,
                    declared_outputs: Optional[Dict[str, str]] = None) -> tuple:
    """Compile body. Returns (last_id, local_map)."""
    local_map: Dict[str, str] = dict(outer_map or {})
    last = None
    current_values = list(incoming or [])
    _consume = consume_incoming  # True only when inlining a motif via USE

    for stmt in body:
        if isinstance(stmt, InputStmt):
            # Consume pipeline incoming only once, for the first INPUT of an
            # inlined motif body. Subsequent INPUTs are normal external inputs.
            if _consume and current_values:
                nid = current_values[-1]
                local_map[stmt.identifier] = nid
                last = nid
                _consume = False  # only the first INPUT of the motif
            else:
                nid = next_id("input")
                graph.nodes[nid] = GraphNode(
                    id=nid, op="input",
                    args=[stmt.identifier],
                    meta={"structure": parent_name, "type": stmt.type_name}
                )
                local_map[stmt.identifier] = nid
                current_values = [nid]
                last = nid
                graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, ConstStmt):
            nid = next_id("const")
            graph.nodes[nid] = GraphNode(
                id=nid, op="const",
                args=[stmt.value],
                meta={"structure": parent_name, "name": stmt.identifier}
            )
            local_map[stmt.identifier] = nid
            last = nid
            graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, LetStmt):
            if stmt.operator is None:
                # simple binding / rename
                src = stmt.args[0] if stmt.args else None
                if isinstance(src, str) and src in local_map:
                    local_map[stmt.identifier] = local_map[src]
                    last = local_map[src]
                elif isinstance(src, (int, float)):
                    nid = next_id("const")
                    graph.nodes[nid] = GraphNode(id=nid, op="const", args=[src],
                                                 meta={"structure": parent_name})
                    local_map[stmt.identifier] = nid
                    last = nid
                    graph.structure_nodes.setdefault(parent_name, []).append(nid)
                else:
                    raise CompileError(f"Cannot resolve LET {stmt.identifier} = {src}")
            else:
                # operator call
                opname = stmt.operator.upper()
                resolved_args = []
                dep_nodes = []
                for a in stmt.args:
                    kind, val = _resolve_arg(a, local_map)
                    if kind == "ref":
                        resolved_args.append(("ref", val))
                        dep_nodes.append(val)
                    else:
                        resolved_args.append(("literal", val))

                nid = next_id(opname.lower())
                graph.nodes[nid] = GraphNode(
                    id=nid, op=opname,
                    args=resolved_args,
                    meta={"structure": parent_name, "bind": stmt.identifier}
                )
                for d in dep_nodes:
                    graph.edges.append((d, nid))
                # if no deps and we have current_values, optionally wire them
                if not dep_nodes and current_values:
                    for cv in current_values:
                        graph.edges.append((cv, nid))
                local_map[stmt.identifier] = nid
                current_values = [nid]
                last = nid
                graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, CallStmt):
            opname = stmt.operator.upper()
            if opname == "IDENTITY" and len(stmt.args) == 1:
                # bare reference
                ref = stmt.args[0]
                if isinstance(ref, str) and ref in local_map:
                    last = local_map[ref]
                    current_values = [last]
                continue

            resolved_args = []
            dep_nodes = []
            for a in stmt.args:
                kind, val = _resolve_arg(a, local_map)
                if kind == "ref":
                    resolved_args.append(("ref", val))
                    dep_nodes.append(val)
                else:
                    resolved_args.append(("literal", val))

            nid = next_id(opname.lower())
            graph.nodes[nid] = GraphNode(
                id=nid, op=opname,
                args=resolved_args,
                meta={"structure": parent_name}
            )
            for d in dep_nodes:
                graph.edges.append((d, nid))
            if not dep_nodes and current_values:
                for cv in current_values:
                    graph.edges.append((cv, nid))
            current_values = [nid]
            last = nid
            graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, TransformStmt):
            # legacy: treat as identity transform label
            nid = next_id("transform")
            graph.nodes[nid] = GraphNode(
                id=nid, op="TRANSFORM",
                args=[("literal", stmt.identifier)],
                meta={"structure": parent_name}
            )
            for cv in current_values:
                graph.edges.append((cv, nid))
            current_values = [nid]
            last = nid
            graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, ConnectStmt):
            nid = next_id("connect")
            graph.nodes[nid] = GraphNode(
                id=nid, op="CONNECT",
                args=[("literal", stmt.identifier)],
                meta={"structure": parent_name}
            )
            for cv in current_values:
                graph.edges.append((cv, nid))
            current_values = [nid]
            last = nid
            graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, EvaluateStmt):
            nid = next_id("evaluate")
            graph.nodes[nid] = GraphNode(
                id=nid, op="EVALUATE",
                args=[("literal", stmt.identifier)],
                meta={"structure": parent_name}
            )
            for cv in current_values:
                graph.edges.append((cv, nid))
            current_values = [nid]
            last = nid
            graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, ObserveStmt):
            nid = next_id("observe")
            graph.nodes[nid] = GraphNode(
                id=nid, op="OBSERVE",
                args=[("literal", stmt.identifier)],
                meta={"structure": parent_name}
            )
            for cv in current_values:
                graph.edges.append((cv, nid))
            last = nid
            graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, AdaptStmt):
            nid = next_id("adapt")
            graph.nodes[nid] = GraphNode(
                id=nid, op="ADAPT",
                args=[("literal", stmt.target)],
                meta={"structure": parent_name}
            )
            for cv in current_values:
                graph.edges.append((cv, nid))
            current_values = [nid]
            last = nid
            graph.structure_nodes.setdefault(parent_name, []).append(nid)

        elif isinstance(stmt, UseStmt):
            if stmt.motif_name not in motifs:
                raise CompileError(f"Unknown MOTIF '{stmt.motif_name}'")
            motif_last, motif_locals = _compile_body(graph, next_id, motifs,
                motifs[stmt.motif_name].body,
                parent_name=f"{parent_name}.{stmt.motif_name}",
                incoming=current_values,
                outer_map=local_map,
                consume_incoming=True,
                declared_outputs=declared_outputs
            )
            # Merge motif locals so outer structure can reference them
            local_map.update(motif_locals)
            if motif_last:
                current_values = [motif_last]
                last = motif_last
                # also bind under the motif name itself
                local_map[stmt.motif_name] = motif_last

        elif isinstance(stmt, StructureOutput):
            # v0.3: named output declaration inside a structure body
            if stmt.name not in local_map:
                raise CompileError(
                    f"OUTPUT '{stmt.name}' refers to unknown identifier "
                    f"in structure '{parent_name}'")
            if declared_outputs is not None:
                if stmt.name in declared_outputs:
                    raise CompileError(
                        f"Duplicate OUTPUT '{stmt.name}' in structure '{parent_name}'")
                declared_outputs[stmt.name] = local_map[stmt.name]

    return last, local_map


def finalize_graph(graph: ComputationalGraph, root_ids: List[str]) -> None:
    """Prune nodes unreachable from the given roots and verify acyclicity."""
    from collections import defaultdict, deque
    parents = defaultdict(list)
    for a, b in graph.edges:
        parents[b].append(a)

    reachable = set()
    q = deque([r for r in root_ids if r])
    while q:
        n = q.popleft()
        if n in reachable:
            continue
        reachable.add(n)
        for p in parents[n]:
            q.append(p)

    graph.nodes = {k: v for k, v in graph.nodes.items() if k in reachable}
    graph.edges = [(a, b) for a, b in graph.edges if a in reachable and b in reachable]

    try:
        graph.topological_order()
    except ValueError as e:
        raise CompileError(str(e))


def compile_program(ast: Genome) -> ComputationalGraph:
    graph = ComputationalGraph(genome_name=ast.name)
    node_counter = [0]

    def next_id(prefix: str) -> str:
        node_counter[0] += 1
        return f"{prefix}_{node_counter[0]}"

    structures: Dict[str, Structure] = {}
    motifs: Dict[str, Motif] = {}
    output_name = None

    for stmt in ast.statements:
        if isinstance(stmt, MemoryDecl):
            graph.memory_kind = stmt.kind
        elif isinstance(stmt, Structure):
            structures[stmt.name] = stmt
        elif isinstance(stmt, Motif):
            motifs[stmt.name] = stmt
        elif isinstance(stmt, OutputStmt):
            output_name = stmt.identifier
        # v0.3: Pipeline statements are ignored by the legacy single-structure
        # path; use the pipeline compiler (pipeline.py) for those genomes.

    if output_name is None:
        raise CompileError("GENOME must declare an OUTPUT")

    # Compile all structures
    for name, struct in structures.items():
        declared: Dict[str, str] = {}
        root, _ = _compile_body(graph, next_id, motifs, struct.body,
                                parent_name=name, declared_outputs=declared)
        if root:
            graph.structure_outputs[name] = root
        if declared:
            graph.structure_declared_outputs[name] = declared

    # Also compile standalone motifs into the motif registry (for later reuse)
    for name, motif in motifs.items():
        graph.motifs[name] = motif  # keep AST for potential re-instantiation

    # Wire the declared OUTPUT
    if output_name in graph.structure_outputs:
        out_id = next_id("output")
        graph.nodes[out_id] = GraphNode(id=out_id, op="output", args=[output_name])
        graph.edges.append((graph.structure_outputs[output_name], out_id))
        graph.output_id = out_id
    elif output_name in structures:
        # structure existed but produced no nodes
        raise CompileError(f"STRUCTURE '{output_name}' produced no computational nodes")
    else:
        raise CompileError(f"OUTPUT '{output_name}' does not refer to a known STRUCTURE")

    finalize_graph(graph, [graph.output_id])

    return graph


def compile_structure_body(structure: Structure, motifs: Dict[str, Motif],
                           genome_name: str,
                           memory_kind: str = "associative") -> tuple:
    """v0.3: compile a single STRUCTURE into its own independent graph.

    Returns (graph, local_map, declared_outputs) where declared_outputs maps
    output name -> node id for every OUTPUT statement in the structure body.
    """
    graph = ComputationalGraph(genome_name=genome_name)
    graph.memory_kind = memory_kind
    node_counter = [0]

    def next_id(prefix: str) -> str:
        node_counter[0] += 1
        return f"{prefix}_{node_counter[0]}"

    declared: Dict[str, str] = {}
    root, local_map = _compile_body(graph, next_id, motifs, structure.body,
                                    parent_name=structure.name,
                                    declared_outputs=declared)
    if declared:
        graph.structure_declared_outputs[structure.name] = declared
    roots = list(declared.values())
    # CONNECT performs a memory-write side effect: retain such nodes even when
    # they are not on the path to a declared output. Without this, pruning
    # would silently discard the write.
    roots += [nid for nid, n in graph.nodes.items() if n.op == "CONNECT"]
    if not roots:
        if root:
            roots = [root]
        else:
            raise CompileError(
                f"STRUCTURE '{structure.name}' produced no computational nodes")
    if root:
        graph.structure_outputs[structure.name] = root
    finalize_graph(graph, roots)
    # Wire declared outputs through `output` nodes so the standard Runtime can
    # execute the graph (Runtime.execute requires graph.output_id). The engine
    # reads declared outputs from runtime.values either way.
    first_out = None
    for oname, nid in declared.items():
        out_id = next_id(f"output_{oname}")
        graph.nodes[out_id] = GraphNode(
            id=out_id, op="output", args=[oname],
            meta={"structure": structure.name, "output_name": oname})
        graph.edges.append((nid, out_id))
        if first_out is None:
            first_out = out_id
    if first_out is not None:
        graph.output_id = first_out
    elif root is not None:
        out_id = next_id("output")
        graph.nodes[out_id] = GraphNode(
            id=out_id, op="output", args=[structure.name],
            meta={"structure": structure.name})
        graph.edges.append((root, out_id))
        graph.output_id = out_id
    return graph, local_map, declared
