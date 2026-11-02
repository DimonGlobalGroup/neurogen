"""
NEUROGEN Intelligence Interface v0.2
"""

from typing import Any, Dict, Optional
from parser import parse, ParseError
from compiler import compile_program, CompileError
from runtime import Runtime
from evolution import ControlledEvolver
from ast_nodes import ComputationalGraph
from operators import list_operators

class Neurogen:
    """
    Primary entry point for NEUROGEN programs.
    """

    def __init__(self):
        self.source: Optional[str] = None
        self.ast = None
        self.graph: Optional[ComputationalGraph] = None
        self.runtime: Optional[Runtime] = None
        self.evolver = ControlledEvolver()
        self.last_result = None

    def load(self, source: str) -> "Neurogen":
        self.source = source
        self.ast = parse(source)
        self.graph = compile_program(self.ast)
        self.runtime = Runtime(self.graph)
        return self

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        if self.runtime is None:
            raise RuntimeError("No program loaded. Call load() first.")
        self.last_result = self.runtime.execute(inputs or {})
        return self.last_result

    def evolve(self, intensity: int = 1) -> "Neurogen":
        if self.graph is None:
            raise RuntimeError("No graph to evolve.")
        self.graph = self.evolver.mutate(self.graph, intensity=intensity)
        self.runtime = Runtime(self.graph)
        return self

    def inspect(self) -> Dict[str, Any]:
        if self.graph is None:
            return {"status": "empty"}
        return {
            "genome": self.graph.genome_name,
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "output": self.graph.output_id,
            "memory_kind": self.graph.memory_kind,
            "node_ops": {nid: n.op for nid, n in self.graph.nodes.items()},
            "structure_outputs": dict(self.graph.structure_outputs),
            "trace": self.runtime.get_trace() if self.runtime else [],
            "memory": self.runtime.get_memory_dump() if self.runtime else [],
            "mutation_log": self.evolver.get_log(),
            "available_operators": list_operators(),
        }

    def source_of_truth(self) -> str:
        return self.source or ""


# ---------------------------------------------------------------------------
# v0.3 pipeline interface
# ---------------------------------------------------------------------------

from ast_nodes import Pipeline, PipelineIR
from structure import CompiledStructure, compile_structure
from pipeline import PipelineCompiler, PipelineLimits
from pipeline_engine import PipelineEngine, PipelineResult, EngineSettings


class NeurogenPipeline:
    """
    Primary entry point for NEUROGEN v0.3 cross-structure pipelines.

    Parses a genome, independently compiles every STRUCTURE, compiles every
    PIPELINE declaration, and executes pipelines through the pipeline engine.
    The legacy Neurogen class above is untouched.
    """

    def __init__(self):
        self.source: Optional[str] = None
        self.ast = None
        self.structures: Dict[str, CompiledStructure] = {}
        self.pipelines: Dict[str, PipelineIR] = {}
        self._compiler: Optional[PipelineCompiler] = None
        # persistent engines per (pipeline, settings): the execution cache
        # lives on the engine, so repeated runs benefit from caching and
        # invalidation is driven by input fingerprints.
        self._engines: Dict[Any, PipelineEngine] = {}

    def load(self, source: str) -> "NeurogenPipeline":
        self.source = source
        self.ast = parse(source)
        self.structures = {}
        for stmt in self.ast.statements:
            if type(stmt).__name__ == "Structure":
                self.structures[stmt.name] = compile_structure(self.ast, stmt.name)
        self._compiler = PipelineCompiler(self.ast, dict(self.structures))
        self.pipelines = self._compiler.compile_all()
        return self

    def pipeline_names(self):
        return sorted(self.pipelines.keys())

    def run(self, pipeline_name: str,
            inputs: Optional[Dict[Any, Any]] = None,
            settings: Optional[EngineSettings] = None) -> PipelineResult:
        if pipeline_name not in self.pipelines:
            raise KeyError(f"Unknown PIPELINE '{pipeline_name}' "
                           f"(available: {self.pipeline_names()})")
        skey = (settings.fingerprint() + "|" + str(settings.cache_enabled)
                if settings else "default")
        key = (pipeline_name, skey)
        engine = self._engines.get(key)
        if engine is None:
            engine = PipelineEngine(self.pipelines[pipeline_name], settings)
            self._engines[key] = engine
        return engine.execute(inputs or {})

    def run_structure(self, structure_name: str,
                      inputs: Optional[Dict[str, Any]] = None,
                      timeout_s: float = 5.0) -> Dict[str, Any]:
        """Execute a single STRUCTURE independently (outside any pipeline)."""
        if structure_name not in self.structures:
            raise KeyError(f"Unknown STRUCTURE '{structure_name}'")
        cs = self.structures[structure_name]
        rt = Runtime(cs.graph, timeout_s=timeout_s)
        result = rt.execute(dict(inputs or {}))
        return {
            "result": result,
            "outputs": {oname: rt.values[nid].to_python()
                        for oname, nid in cs.outputs.items()},
            "trace": rt.get_trace(),
            "memory": rt.get_memory_dump(),
        }

    def inspect_pipeline(self, name: str) -> Dict[str, Any]:
        if name not in self.pipelines:
            raise KeyError(f"Unknown PIPELINE '{name}'")
        ir = self.pipelines[name]
        return {
            "pipeline": ir.name,
            "structures": {
                sname: {
                    "inputs": cs.inputs,
                    "outputs": {o: cs.output_dtypes[o].name
                                for o in cs.outputs},
                    "nodes": len(cs.graph.nodes),
                    "has_memory_write": cs.has_memory_write,
                }
                for sname, cs in ir.structures.items()
            },
            "connections": [
                {"from": f"{c.src_struct}.{c.src_output}",
                 "to": f"{c.dst_struct}.{c.dst_input}",
                 "types": f"{c.src_dtype.name}->{c.dst_dtype.name}"}
                for c in ir.connections
            ],
            "exec_order": ir.exec_order,
            "outputs": [f"{s}.{o}" for s, o in ir.outputs],
            "meta": ir.meta,
        }
