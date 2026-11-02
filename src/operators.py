"""
NEUROGEN Operator Library v0.2 — real computational semantics.
"""

from __future__ import annotations
from typing import List, Callable, Dict, Any, Optional, Tuple
import numpy as np
from neuro_types import NeuroValue, DType, promote_type, NUMERIC, SCALAR


class OperatorError(Exception):
    pass


class Operator:
    def __init__(
        self,
        name: str,
        arity: int,
        input_types: List[set],
        output_type: DType,
        fn: Callable,
        description: str = "",
    ):
        self.name = name
        self.arity = arity
        self.input_types = input_types
        self.output_type = output_type
        self.fn = fn
        self.description = description

    def validate(self, args: List[NeuroValue]) -> None:
        if len(args) != self.arity and self.arity >= 0:
            raise OperatorError(f"{self.name}: expected {self.arity} args, got {len(args)}")
        for i, (arg, allowed) in enumerate(zip(args, self.input_types)):
            if DType.ANY not in allowed and arg.dtype not in allowed and not (
                arg.dtype in NUMERIC and any(t in NUMERIC for t in allowed)
            ):
                raise OperatorError(
                    f"{self.name}: arg {i} has type {arg.dtype.name}, expected one of {[t.name for t in allowed]}"
                )

    def execute(self, args: List[NeuroValue]) -> NeuroValue:
        self.validate(args)
        try:
            result = self.fn(*args)
            if not isinstance(result, NeuroValue):
                result = NeuroValue.from_python(result)
            return result
        except Exception as e:
            raise OperatorError(f"{self.name} failed: {e}") from e


# ---------- Math helpers ----------

def _binary_numeric(op: Callable):
    def wrapper(a: NeuroValue, b: NeuroValue) -> NeuroValue:
        aa, bb = a.as_array(), b.as_array()
        try:
            out = op(aa, bb)
        except Exception:
            # broadcast attempt
            out = op(np.asarray(aa, dtype=float), np.asarray(bb, dtype=float))
        return NeuroValue.from_python(out)
    return wrapper


def _unary_numeric(op: Callable):
    def wrapper(a: NeuroValue) -> NeuroValue:
        return NeuroValue.from_python(op(a.as_array()))
    return wrapper


# ---------- Operator Registry ----------

REGISTRY: Dict[str, Operator] = {}


def register(op: Operator):
    REGISTRY[op.name.upper()] = op
    return op


# Mathematical
register(Operator("ADD", 2, [NUMERIC, NUMERIC], DType.FLOAT, _binary_numeric(np.add), "Element-wise addition"))
register(Operator("SUBTRACT", 2, [NUMERIC, NUMERIC], DType.FLOAT, _binary_numeric(np.subtract)))
register(Operator("MULTIPLY", 2, [NUMERIC, NUMERIC], DType.FLOAT, _binary_numeric(np.multiply)))
def _divide_op(a: NeuroValue, b: NeuroValue) -> NeuroValue:
    aa = a.as_array().astype(float)
    bb = b.as_array().astype(float)
    if np.any(bb == 0):
        raise OperatorError("DIVIDE by zero")
    return NeuroValue.from_python(np.divide(aa, bb))

register(Operator("DIVIDE", 2, [NUMERIC, NUMERIC], DType.FLOAT, _divide_op))
register(Operator("POWER", 2, [NUMERIC, NUMERIC], DType.FLOAT, _binary_numeric(np.power)))
register(Operator("MODULO", 2, [NUMERIC, NUMERIC], DType.FLOAT, _binary_numeric(np.mod)))
register(Operator("MINIMUM", 2, [NUMERIC, NUMERIC], DType.FLOAT, _binary_numeric(np.minimum)))
register(Operator("MAXIMUM", 2, [NUMERIC, NUMERIC], DType.FLOAT, _binary_numeric(np.maximum)))

# Logical
register(Operator("AND", 2, [SCALAR | {DType.VECTOR}, SCALAR | {DType.VECTOR}], DType.BOOL,
                  lambda a, b: NeuroValue.from_python(np.logical_and(a.as_array(), b.as_array()))))
register(Operator("OR", 2, [SCALAR | {DType.VECTOR}, SCALAR | {DType.VECTOR}], DType.BOOL,
                  lambda a, b: NeuroValue.from_python(np.logical_or(a.as_array(), b.as_array()))))
register(Operator("NOT", 1, [SCALAR | {DType.VECTOR}], DType.BOOL,
                  lambda a: NeuroValue.from_python(np.logical_not(a.as_array()))))
register(Operator("XOR", 2, [SCALAR | {DType.VECTOR}, SCALAR | {DType.VECTOR}], DType.BOOL,
                  lambda a, b: NeuroValue.from_python(np.logical_xor(a.as_array(), b.as_array()))))
register(Operator("EQUAL", 2, [NUMERIC | {DType.BOOL, DType.STRING}, NUMERIC | {DType.BOOL, DType.STRING}], DType.BOOL,
                  lambda a, b: NeuroValue.from_python(np.equal(a.as_array() if a.is_numeric() else a.data,
                                                              b.as_array() if b.is_numeric() else b.data))))
register(Operator("GREATER", 2, [NUMERIC, NUMERIC], DType.BOOL,
                  lambda a, b: NeuroValue.from_python(np.greater(a.as_array(), b.as_array()))))
register(Operator("LESS", 2, [NUMERIC, NUMERIC], DType.BOOL,
                  lambda a, b: NeuroValue.from_python(np.less(a.as_array(), b.as_array()))))

# Numerical transforms
register(Operator("TANH", 1, [NUMERIC], DType.FLOAT, _unary_numeric(np.tanh)))
register(Operator("SIGMOID", 1, [NUMERIC], DType.FLOAT,
                  _unary_numeric(lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))))
register(Operator("RELU", 1, [NUMERIC], DType.FLOAT, _unary_numeric(lambda x: np.maximum(0, x))))
register(Operator("SOFTMAX", 1, [NUMERIC], DType.VECTOR,
                  lambda a: NeuroValue.from_python(
                      (lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum())(a.as_array().astype(float))
                  )))
register(Operator("NORMALIZE", 1, [NUMERIC], DType.FLOAT,
                  lambda a: NeuroValue.from_python(
                      (lambda x: (x - x.mean()) / (x.std() + 1e-8))(a.as_array().astype(float))
                  )))
register(Operator("CLAMP", 3, [NUMERIC, NUMERIC, NUMERIC], DType.FLOAT,
                  lambda a, lo, hi: NeuroValue.from_python(np.clip(a.as_array(), lo.as_array(), hi.as_array()))))

# Data operations
register(Operator("VECTOR", -1, [], DType.VECTOR,
                  lambda *args: NeuroValue.from_python([x.to_python() if isinstance(x, NeuroValue) else x for x in args])))
register(Operator("MATRIX", -1, [], DType.MATRIX,
                  lambda *args: NeuroValue.from_python([x.to_python() if isinstance(x, NeuroValue) else x for x in args])))
register(Operator("INDEX", 2, [NUMERIC, {DType.INT, DType.FLOAT}], DType.ANY,
                  lambda a, idx: NeuroValue.from_python(a.as_array()[int(idx.to_python())])))
register(Operator("SLICE", 3, [NUMERIC, {DType.INT}, {DType.INT}], DType.VECTOR,
                  lambda a, start, end: NeuroValue.from_python(a.as_array()[int(start.to_python()):int(end.to_python())])))
register(Operator("CONCATENATE", 2, [NUMERIC, NUMERIC], DType.VECTOR,
                  lambda a, b: NeuroValue.from_python(np.concatenate([a.as_array().ravel(), b.as_array().ravel()]))))
register(Operator("REDUCE", 1, [NUMERIC], DType.FLOAT,
                  lambda a: NeuroValue.from_python(float(np.sum(a.as_array())))))
register(Operator("MAP", 2, [NUMERIC, {DType.STRING}], DType.VECTOR,  # simplified: only named transforms
                  lambda a, name: REGISTRY[name.data.upper()].execute([a]) if name.data.upper() in REGISTRY
                  else (_ for _ in ()).throw(OperatorError(f"Unknown map target {name.data}"))))

# Computational / structural (still meaningful)
register(Operator("TRANSFORM", 2, [{DType.ANY}, {DType.STRING}], DType.ANY,
                  lambda data, label: NeuroValue(
                      {"transformed": label.to_python(), "data": data.to_python()},
                      DType.OBJECT
                  )))
register(Operator("AGGREGATE", 1, [NUMERIC], DType.FLOAT,
                  lambda a: NeuroValue.from_python(float(np.mean(a.as_array())))))
register(Operator("MERGE", 2, [{DType.ANY}, {DType.ANY}], DType.OBJECT,
                  lambda a, b: NeuroValue({"left": a.to_python(), "right": b.to_python()}, DType.OBJECT)))
register(Operator("ROUTE", 2, [{DType.ANY}, {DType.STRING}], DType.ANY,
                  lambda data, dest: NeuroValue({"routed_to": dest.to_python(), "payload": data.to_python()}, DType.OBJECT)))


def get_operator(name: str) -> Operator:
    key = name.upper()
    if key not in REGISTRY:
        raise OperatorError(f"Unknown operator: {name}")
    return REGISTRY[key]


def list_operators() -> List[str]:
    return sorted(REGISTRY.keys())
