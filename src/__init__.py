"""
NEUROGEN — AI-native programming language and computational environment.
"""

from parser import parse
from ast_nodes import *
from compiler import compile_program, compile_structure_body
from runtime import Runtime
from memory import AssociativeMemory
from evolution import ControlledEvolver
from interface import Neurogen, NeurogenPipeline
from operators import list_operators, get_operator
from neuro_types import NeuroValue, DType
from structure import compile_structure, CompiledStructure
from pipeline import PipelineCompiler
from pipeline_engine import PipelineEngine, PipelineResult, EngineSettings
from errors import (NeurogenError, StructureError, ConnectionError,
                    PipelineError, StructureInputError, StructureExecutionError)

__version__ = "0.3.0"
__all__ = [
    "parse",
    "compile_program",
    "compile_structure_body",
    "Runtime",
    "AssociativeMemory",
    "ControlledEvolver",
    "Neurogen",
    "NeurogenPipeline",
    "compile_structure",
    "CompiledStructure",
    "PipelineCompiler",
    "PipelineEngine",
    "PipelineResult",
    "EngineSettings",
    "NeurogenError",
    "StructureError",
    "ConnectionError",
    "PipelineError",
    "StructureInputError",
    "StructureExecutionError",
    "list_operators",
    "get_operator",
    "NeuroValue",
    "DType",
]
