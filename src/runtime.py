"""
NEUROGEN Runtime v0.2 — executes real computational graphs with operators.
"""

from typing import Any, Dict, Optional, List
import time
from ast_nodes import ComputationalGraph, GraphNode
from memory import AssociativeMemory, KeyValueMemory
from neuro_types import NeuroValue, DType
from operators import get_operator, OperatorError, REGISTRY


class RuntimeError(Exception):
    pass


class Runtime:
    def __init__(self, graph: ComputationalGraph, max_steps: int = 10000, timeout_s: float = 5.0):
        self.graph = graph
        if graph.memory_kind == "associative":
            self.memory = AssociativeMemory()
        else:
            self.memory = KeyValueMemory()
        self.context: Dict[str, Any] = {}
        self.trace: List[dict] = []
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.values: Dict[str, NeuroValue] = {}

    def set_input(self, name: str, value: Any):
        self.context[name] = value

    def execute(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        if inputs:
            self.context.update(inputs)

        self.values.clear()
        self.trace.clear()
        start = time.time()
        order = self.graph.topological_order()

        if len(order) > self.max_steps:
            raise RuntimeError(f"Graph too large: {len(order)} nodes (limit {self.max_steps})")

        for i, nid in enumerate(order):
            if time.time() - start > self.timeout_s:
                raise RuntimeError(f"Execution timeout after {self.timeout_s}s")
            node = self.graph.nodes[nid]
            result = self._eval_node(node)
            self.values[nid] = result
            self.trace.append({
                "node": nid,
                "op": node.op,
                "result_type": result.dtype.name if isinstance(result, NeuroValue) else type(result).__name__,
                "result": result.to_python() if isinstance(result, NeuroValue) else result,
            })

        if self.graph.output_id is None:
            raise RuntimeError("No output node in graph")
        out = self.values.get(self.graph.output_id)
        if isinstance(out, NeuroValue):
            return out.to_python()
        return out

    def _resolve_arg(self, arg) -> NeuroValue:
        if isinstance(arg, tuple) and len(arg) == 2:
            kind, val = arg
            if kind == "ref":
                if val not in self.values:
                    raise RuntimeError(f"Unresolved reference to node {val}")
                return self.values[val]
            else:  # literal
                return NeuroValue.from_python(val)
        # legacy plain values
        return NeuroValue.from_python(arg)

    def _eval_node(self, node: GraphNode) -> NeuroValue:
        op = node.op

        if op == "input":
            name = node.args[0]
            expected = node.meta.get("type")
            if name not in self.context:
                raise RuntimeError(f"Missing required input: '{name}'")
            val = NeuroValue.from_python(self.context[name])
            if expected:
                expected = expected.upper()
                type_map = {
                    "BOOL": DType.BOOL, "INT": DType.INT, "FLOAT": DType.FLOAT,
                    "STRING": DType.STRING, "VECTOR": DType.VECTOR,
                    "MATRIX": DType.MATRIX, "TENSOR": DType.TENSOR, "OBJECT": DType.OBJECT,
                }
                want = type_map.get(expected)
                if want and val.dtype != want:
                    # allow INT->FLOAT promotion
                    if not (want == DType.FLOAT and val.dtype == DType.INT):
                        raise RuntimeError(
                            f"Input '{name}': expected {expected}, got {val.dtype.name}"
                        )
            return val

        if op == "const":
            return NeuroValue.from_python(node.args[0])

        if op == "output":
            # forward the single predecessor
            preds = [a for a, b in self.graph.edges if b == node.id]
            if not preds:
                return NeuroValue(None, DType.OBJECT)
            return self.values[preds[0]]

        # Special structural ops that still do useful work
        if op == "CONNECT":
            preds = [self.values[a] for a, b in self.graph.edges if b == node.id]
            target = node.args[0][1] if isinstance(node.args[0], tuple) else node.args[0]
            payload = preds[0] if preds else NeuroValue(None, DType.OBJECT)
            py_val = payload.to_python()
            if isinstance(self.memory, AssociativeMemory):
                self.memory.store(key=str(target), value=py_val, tags=["connection", str(target)])
            elif isinstance(self.memory, KeyValueMemory):
                self.memory.set(str(target), py_val)
            return NeuroValue({"connected_to": target, "data": py_val}, DType.OBJECT)

        if op == "TRANSFORM":
            preds = [self.values[a] for a, b in self.graph.edges if b == node.id]
            label = node.args[0][1] if node.args and isinstance(node.args[0], tuple) else (node.args[0] if node.args else "transform")
            data = preds[0].to_python() if preds else None
            return NeuroValue({"transformed": label, "data": data}, DType.OBJECT)

        if op == "EVALUATE":
            preds = [self.values[a] for a, b in self.graph.edges if b == node.id]
            what = node.args[0][1] if node.args and isinstance(node.args[0], tuple) else "eval"
            return NeuroValue({"evaluated": what, "on": [p.to_python() for p in preds]}, DType.OBJECT)

        if op == "OBSERVE":
            preds = [self.values[a] for a, b in self.graph.edges if b == node.id]
            target = node.args[0][1] if node.args and isinstance(node.args[0], tuple) else "observe"
            return NeuroValue({"observed": target, "state": [p.to_python() for p in preds]}, DType.OBJECT)

        if op == "ADAPT":
            preds = [self.values[a] for a, b in self.graph.edges if b == node.id]
            target = node.args[0][1] if node.args and isinstance(node.args[0], tuple) else "adapt"
            return NeuroValue({"adapted": target, "based_on": [p.to_python() for p in preds]}, DType.OBJECT)

        if op == "IDENTITY":
            preds = [self.values[a] for a, b in self.graph.edges if b == node.id]
            return preds[0] if preds else NeuroValue(None, DType.OBJECT)

        # Real operators from the registry
        if op in REGISTRY:
            op_obj = get_operator(op)
            arg_vals = []
            if node.args:
                for a in node.args:
                    arg_vals.append(self._resolve_arg(a))
            # If arity does not match, prefer predecessors
            preds = [a for a, b in self.graph.edges if b == node.id]
            if (not arg_vals or (op_obj.arity >= 0 and len(arg_vals) != op_obj.arity)) and preds:
                arg_vals = [self.values[p] for p in preds]
            # Truncate or pad to arity if needed for unary ops
            if op_obj.arity == 1 and len(arg_vals) > 1:
                arg_vals = arg_vals[:1]
            try:
                return op_obj.execute(arg_vals)
            except OperatorError as e:
                raise RuntimeError(str(e)) from e

        raise RuntimeError(f"Unknown op: {op}")

    def get_trace(self):
        return self.trace

    def get_memory_dump(self):
        return self.memory.dump()
