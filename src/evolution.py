"""
NEUROGEN Controlled Evolution — validity-preserving mutations.
"""

import copy
import random
from typing import List, Optional, Tuple
from ast_nodes import ComputationalGraph, GraphNode


class ControlledEvolver:
    """
    Performs controlled, validity-preserving mutations on a computational graph.
    Inspired by the rooted typed graph findings from the earlier experiments.
    """

    def __init__(self, max_nodes: int = 32):
        self.max_nodes = max_nodes
        self.mutation_log: List[str] = []

    def mutate(self, graph: ComputationalGraph, intensity: int = 1) -> ComputationalGraph:
        """
        Apply 1..intensity mutations that preserve:
        - rootedness
        - no cycles
        - every node contributes to the output
        """
        g = copy.deepcopy(graph)
        for _ in range(intensity):
            choice = random.choice([
                self._add_transform,
                self._duplicate_node,
                self._prune_redundant,
            ])
            try:
                choice(g)
            except Exception as e:
                self.mutation_log.append(f"Mutation failed: {e}")
                continue
        # Always re-validate
        self._ensure_valid(g)
        return g

    def _add_transform(self, g: ComputationalGraph):
        if len(g.nodes) >= self.max_nodes:
            return
        # Pick a random non-output node and insert a transform after it
        candidates = [nid for nid, n in g.nodes.items() if n.op != "output"]
        if not candidates:
            return
        src = random.choice(candidates)
        new_id = f"mut_relu_{len(g.nodes)+1}"
        g.nodes[new_id] = GraphNode(id=new_id, op="RELU", args=[])
        # Redirect some outgoing edges through the new node
        new_edges = []
        redirected = False
        for a, b in g.edges:
            if a == src and not redirected:
                new_edges.append((a, new_id))
                new_edges.append((new_id, b))
                redirected = True
            else:
                new_edges.append((a, b))
        if not redirected:
            # just attach it somewhere that reaches output
            new_edges.append((src, new_id))
            # connect to output if possible
            if g.output_id:
                new_edges.append((new_id, g.output_id))
        g.edges = new_edges
        self.mutation_log.append(f"Added RELU {new_id} after {src}")

    def _duplicate_node(self, g: ComputationalGraph):
        candidates = [nid for nid, n in g.nodes.items() if n.op not in ("output", "input")]
        if not candidates or len(g.nodes) >= self.max_nodes:
            return
        src = random.choice(candidates)
        orig = g.nodes[src]
        new_id = f"dup_{src}_{len(g.nodes)+1}"
        g.nodes[new_id] = GraphNode(id=new_id, op=orig.op, args=list(orig.args), meta=dict(orig.meta))
        # Connect it in parallel from the same parents
        parents = [a for a, b in g.edges if b == src]
        for p in parents:
            g.edges.append((p, new_id))
        # And feed into the same children
        children = [b for a, b in g.edges if a == src]
        for c in children:
            g.edges.append((new_id, c))
        self.mutation_log.append(f"Duplicated {src} → {new_id}")

    def _prune_redundant(self, g: ComputationalGraph):
        # Very simple: remove a non-critical transform if it has only one parent and one child
        for nid, node in list(g.nodes.items()):
            if node.op != "transform":
                continue
            parents = [a for a, b in g.edges if b == nid]
            children = [b for a, b in g.edges if a == nid]
            if len(parents) == 1 and len(children) == 1:
                # bypass
                g.edges = [(a, b) for a, b in g.edges if a != nid and b != nid]
                g.edges.append((parents[0], children[0]))
                del g.nodes[nid]
                self.mutation_log.append(f"Pruned redundant {nid}")
                return

    def _rewire(self, g: ComputationalGraph):
        # Randomly change one edge target (with care)
        if len(g.edges) < 2:
            return
        idx = random.randint(0, len(g.edges) - 1)
        a, b = g.edges[idx]
        candidates = [nid for nid in g.nodes if nid != a and nid != b and g.nodes[nid].op != "input"]
        if not candidates:
            return
        new_b = random.choice(candidates)
        g.edges[idx] = (a, new_b)
        self.mutation_log.append(f"Rewired {a} → {b} became {a} → {new_b}")

    def _ensure_valid(self, g: ComputationalGraph):
        # Re-prune unreachable and check for cycles
        from collections import defaultdict, deque
        parents = defaultdict(list)
        for a, b in g.edges:
            parents[b].append(a)

        reachable = set()
        if g.output_id:
            q = deque([g.output_id])
            while q:
                n = q.popleft()
                if n in reachable:
                    continue
                reachable.add(n)
                for p in parents[n]:
                    q.append(p)

        g.nodes = {k: v for k, v in g.nodes.items() if k in reachable}
        g.edges = [(a, b) for a, b in g.edges if a in reachable and b in reachable]

        # topological check
        g.topological_order()

    def get_log(self) -> List[str]:
        return self.mutation_log
