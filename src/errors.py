"""
NEUROGEN v0.3 — Error hierarchy for structures and pipelines.

New error classes subclass both the common NeurogenError and the relevant
v0.2.1 error class, so existing `except CompileError` / `except RuntimeError`
handlers keep working while new code can catch the precise class.
"""

from compiler import CompileError
from runtime import RuntimeError as NGRuntimeError


class NeurogenError(Exception):
    """Common base for all NEUROGEN v0.3 errors."""


class StructureError(NeurogenError, CompileError):
    """Structure compilation / interface errors (unknown structure, bad OUTPUT)."""


class ConnectionError(NeurogenError, CompileError):
    """Connection validation errors (unknown endpoint, type mismatch, duplicate)."""


class PipelineError(NeurogenError, CompileError):
    """Pipeline-level errors (cycles, unreachable structures, budget violations)."""


class StructureInputError(NeurogenError, NGRuntimeError):
    """Runtime input errors: missing input or failed runtime type validation."""


class StructureExecutionError(NeurogenError, NGRuntimeError):
    """Runtime execution errors inside a pipeline structure."""
