"""
NEUROGEN Type System and Value Representation (v0.2)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Tuple
from enum import Enum, auto
import numpy as np


class DType(Enum):
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    VECTOR = auto()
    MATRIX = auto()
    TENSOR = auto()
    OBJECT = auto()
    GRAPH_REF = auto()
    MEMORY_REF = auto()
    ANY = auto()


@dataclass
class NeuroValue:
    """Unified runtime value with type information."""
    data: Any
    dtype: DType
    shape: Optional[Tuple[int, ...]] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.shape is None and isinstance(self.data, np.ndarray):
            self.shape = self.data.shape
        elif self.shape is None and isinstance(self.data, (list, tuple)):
            try:
                arr = np.asarray(self.data)
                self.shape = arr.shape
                if self.dtype in (DType.VECTOR, DType.MATRIX, DType.TENSOR):
                    self.data = arr
            except Exception:
                pass

    @staticmethod
    def from_python(val: Any) -> "NeuroValue":
        if isinstance(val, NeuroValue):
            return val
        if isinstance(val, (bool, np.bool_)):
            return NeuroValue(bool(val), DType.BOOL)
        if isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)):
            return NeuroValue(int(val), DType.INT)
        if isinstance(val, (float, np.floating)):
            return NeuroValue(float(val), DType.FLOAT)
        if isinstance(val, str):
            return NeuroValue(val, DType.STRING)
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                return NeuroValue(val, DType.VECTOR, val.shape)
            if val.ndim == 2:
                return NeuroValue(val, DType.MATRIX, val.shape)
            return NeuroValue(val, DType.TENSOR, val.shape)
        if isinstance(val, (list, tuple)):
            try:
                arr = np.asarray(val, dtype=float)
                if arr.ndim == 1:
                    return NeuroValue(arr, DType.VECTOR, arr.shape)
                if arr.ndim == 2:
                    return NeuroValue(arr, DType.MATRIX, arr.shape)
                return NeuroValue(arr, DType.TENSOR, arr.shape)
            except Exception:
                return NeuroValue(val, DType.OBJECT)
        return NeuroValue(val, DType.OBJECT)

    def to_python(self) -> Any:
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 0:
                return self.data.item()
            return self.data.tolist()
        return self.data

    def is_numeric(self) -> bool:
        return self.dtype in (DType.INT, DType.FLOAT, DType.VECTOR, DType.MATRIX, DType.TENSOR)

    def as_array(self) -> np.ndarray:
        if isinstance(self.data, np.ndarray):
            return self.data
        if self.dtype in (DType.INT, DType.FLOAT, DType.BOOL):
            return np.asarray(self.data)
        raise TypeError(f"Cannot convert {self.dtype} to array")

    def __repr__(self):
        return f"NeuroValue({self.dtype.name}, shape={self.shape}, data={self.data!r})"


# Type compatibility helpers
NUMERIC = {DType.INT, DType.FLOAT, DType.VECTOR, DType.MATRIX, DType.TENSOR}
SCALAR = {DType.INT, DType.FLOAT, DType.BOOL}


def promote_type(a: DType, b: DType) -> DType:
    if a == b:
        return a
    if a == DType.ANY or b == DType.ANY:
        return DType.ANY
    if a in NUMERIC and b in NUMERIC:
        if DType.TENSOR in (a, b):
            return DType.TENSOR
        if DType.MATRIX in (a, b):
            return DType.MATRIX
        if DType.VECTOR in (a, b):
            return DType.VECTOR
        if DType.FLOAT in (a, b):
            return DType.FLOAT
        return DType.INT
    return DType.OBJECT
