"""
NEUROGEN v0.3 — Pipeline execution engine.

Executes a compiled PipelineIR: structures run in topological order, each in
its own Runtime, with real NeuroValue transport between stages, typed
validation, copy-on-fan-out branching, caching, failure isolation, and
structured tracing.
"""

from __future__ import annotations
import copy
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ast_nodes import PipelineIR, Connection
from structure import CompiledStructure, dtype_from_name
from runtime import Runtime
from neuro_types import NeuroValue, DType
from errors import (
    NeurogenError, PipelineError, StructureInputError, StructureExecutionError,
)


# dtypes whose values may be shared between branches without copying
_IMMUTABLE_DTYPES = {DType.BOOL, DType.INT, DType.FLOAT}


def fanout_value(value: NeuroValue) -> NeuroValue:
    """Copy-on-fan-out (refinement 1).

    Scalar NeuroValues (BOOL/INT/FLOAT) are immutable and may be shared.
    VECTOR/MATRIX/TENSOR (numpy-backed) and OBJECT payloads are deep-copied
    so that branches can never share writable memory.
    """
    if not isinstance(value, NeuroValue):
        return value
    if value.dtype in _IMMUTABLE_DTYPES:
        return value
    return copy.deepcopy(value)


def _runtime_compatible(actual: DType, expected: DType) -> bool:
    """Runtime value check: exact match, INT->FLOAT promotion, or ANY."""
    if actual == expected:
        return True
    if expected == DType.ANY or actual == DType.ANY:
        return True
    if actual == DType.INT and expected == DType.FLOAT:
        return True
    return False


def _fingerprint_value(value: NeuroValue) -> str:
    import numpy as np
    if not isinstance(value, NeuroValue):
        value = NeuroValue.from_python(value)
    data = value.data
    if isinstance(data, np.ndarray):
        dh = hashlib.sha256(data.tobytes()).hexdigest()
    else:
        dh = hashlib.sha256(repr(data).encode("utf-8")).hexdigest()
    return f"{value.dtype.name}:{value.shape}:{dh}"


def summarize_value(value: NeuroValue) -> str:
    """Short human-readable summary (no tensor dumps)."""
    if not isinstance(value, NeuroValue):
        return type(value).__name__
    if value.dtype in _IMMUTABLE_DTYPES:
        return f"{value.dtype.name}({value.data!r})"
    return f"{value.dtype.name}{value.shape}"


@dataclass
class EngineSettings:
    timeout_s: float = 5.0
    max_steps: int = 10000
    cache_enabled: bool = True
    total_budget_s: float = 60.0

    def fingerprint(self) -> str:
        return f"{self.timeout_s}:{self.max_steps}"


@dataclass
class StructureTraceEntry:
    order: int
    name: str
    status: str  # success | failed | blocked
    inputs_summary: Dict[str, str] = field(default_factory=dict)
    outputs_summary: Dict[str, str] = field(default_factory=dict)
    cache: str = "disabled"  # hit | miss | bypass | disabled
    error: Optional[Dict[str, str]] = None
    duration_s: float = 0.0
    memory: Any = None


@dataclass
class PipelineResult:
    pipeline_name: str
    execution_id: str
    status: str  # success | failed
    structure_status: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    trace: List[StructureTraceEntry] = field(default_factory=list)
    connections_traversed: List[Tuple[str, str, str, str]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    duration_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class PipelineEngine:
    def __init__(self, ir: PipelineIR, settings: Optional[EngineSettings] = None):
        self.ir = ir
        self.settings = settings or EngineSettings()
        self._cache: Dict[str, Dict[str, NeuroValue]] = {}
        # upstream map: dst_struct -> list of (src_struct, src_output, dst_input)
        self._incoming: Dict[str, List[Connection]] = {}
        for c in ir.connections:
            self._incoming.setdefault(c.dst_struct, []).append(c)
        self._deps: Dict[str, List[str]] = {}
        for a, b in ir.dep_edges:
            self._deps.setdefault(b, []).append(a)

    # ---- public API ----

    def execute(self, inputs: Optional[Dict[Any, Any]] = None) -> PipelineResult:
        t_start = time.perf_counter()
        execution_id = uuid.uuid4().hex[:12]
        norm_inputs = self._normalize_inputs(inputs or {})
        statuses: Dict[str, str] = {}
        values: Dict[Tuple[str, str], NeuroValue] = {}
        trace: List[StructureTraceEntry] = []
        errors: List[Dict[str, str]] = []
        traversed: List[Tuple[str, str, str, str]] = []
        cache_hits = 0
        cache_misses = 0

        for order, sname in enumerate(self.ir.exec_order):
            if time.perf_counter() - t_start > self.settings.total_budget_s:
                raise PipelineError(
                    f"PIPELINE '{self.ir.name}': total execution budget "
                    f"({self.settings.total_budget_s}s) exceeded")
            cs = self.ir.structures[sname]
            entry = StructureTraceEntry(order=order, name=sname, status="success")

            # failure isolation: blocked if any upstream failed/blocked
            blockers = [d for d in self._deps.get(sname, [])
                        if statuses.get(d) in ("failed", "blocked")]
            if blockers:
                entry.status = "blocked"
                entry.error = {"class": "Blocked",
                               "structure": sname,
                               "message": f"blocked by upstream failure: {blockers}"}
                statuses[sname] = "blocked"
                trace.append(entry)
                continue

            try:
                struct_inputs = self._gather_inputs(cs, values, norm_inputs, traversed)
                entry.inputs_summary = {k: summarize_value(v)
                                        for k, v in struct_inputs.items()}
                out_vals, cache_state, mem_dump, dur = self._run_structure(
                    cs, struct_inputs)
                entry.cache = cache_state
                entry.duration_s = dur
                entry.memory = mem_dump
                if cache_state == "hit":
                    cache_hits += 1
                elif cache_state == "miss":
                    cache_misses += 1
                for oname, nv in out_vals.items():
                    values[(sname, oname)] = nv
                entry.outputs_summary = {k: summarize_value(v)
                                         for k, v in out_vals.items()}
                statuses[sname] = "success"
            except (StructureInputError, StructureExecutionError) as e:
                entry.status = "failed"
                entry.error = {"class": type(e).__name__, "structure": sname,
                               "message": str(e)}
                statuses[sname] = "failed"
                errors.append(entry.error)
            except Exception as e:  # never let an unexpected error escape unmarked
                entry.status = "failed"
                entry.error = {"class": type(e).__name__, "structure": sname,
                               "message": f"unexpected: {e}"}
                statuses[sname] = "failed"
                errors.append(entry.error)
            trace.append(entry)

        outputs: Dict[str, Any] = {}
        ok = True
        for sname, oname in self.ir.outputs:
            if statuses.get(sname) == "success" and (sname, oname) in values:
                outputs[f"{sname}.{oname}"] = values[(sname, oname)].to_python()
            else:
                ok = False
        if any(s == "failed" for s in statuses.values()):
            ok = False

        return PipelineResult(
            pipeline_name=self.ir.name,
            execution_id=execution_id,
            status="success" if ok else "failed",
            structure_status=statuses,
            outputs=outputs,
            trace=trace,
            connections_traversed=traversed,
            errors=errors,
            duration_s=time.perf_counter() - t_start,
            meta={"cache_hits": cache_hits, "cache_misses": cache_misses,
                  "num_structures": len(self.ir.exec_order)},
        )

    # ---- internals ----

    def _normalize_inputs(self, inputs: Dict[Any, Any]) -> Dict[Tuple[str, str], Any]:
        """Accept 'struct.input' strings or (struct, input) tuples; resolve
        bare input names when unambiguous across entry structures."""
        norm: Dict[Tuple[str, str], Any] = {}
        connected_inputs = {(c.dst_struct, c.dst_input) for c in self.ir.connections}
        # candidate bare names: unconnected declared inputs
        bare: Dict[str, List[str]] = {}
        for sname, cs in self.ir.structures.items():
            for iname, _ in cs.inputs:
                if (sname, iname) not in connected_inputs:
                    bare.setdefault(iname, []).append(sname)
        for key, val in inputs.items():
            if isinstance(key, tuple) and len(key) == 2:
                norm[(key[0], key[1])] = val
            elif isinstance(key, str) and "." in key:
                s, _, i = key.partition(".")
                norm[(s, i)] = val
            elif isinstance(key, str):
                owners = bare.get(key, [])
                if len(owners) == 1:
                    norm[(owners[0], key)] = val
                elif not owners:
                    raise StructureInputError(
                        f"Unknown input '{key}': no structure declares it")
                else:
                    raise StructureInputError(
                        f"Ambiguous input '{key}': declared by {owners}; "
                        f"use 'struct.input' form")
            else:
                raise StructureInputError(f"Invalid input key: {key!r}")
        return norm

    def _gather_inputs(self, cs: CompiledStructure,
                       values: Dict[Tuple[str, str], NeuroValue],
                       norm_inputs: Dict[Tuple[str, str], Any],
                       traversed: List[Tuple[str, str, str, str]]
                       ) -> Dict[str, NeuroValue]:
        struct_inputs: Dict[str, NeuroValue] = {}
        fed = {(c.dst_input): c for c in self._incoming.get(cs.name, [])}
        for iname, _tname in cs.inputs:
            if iname in fed:
                c = fed[iname]
                key = (c.src_struct, c.src_output)
                if key not in values:
                    raise StructureInputError(
                        f"STRUCTURE '{cs.name}': input '{iname}' has no value "
                        f"from {c.src_struct}.{c.src_output}")
                nv = fanout_value(values[key])
                # defense in depth: re-validate the actual runtime value
                want = cs.input_dtype(iname)
                if not _runtime_compatible(nv.dtype, want):
                    raise StructureInputError(
                        f"STRUCTURE '{cs.name}': input '{iname}' received "
                        f"{nv.dtype.name}, expected {want.name}")
                struct_inputs[iname] = nv
                traversed.append((c.src_struct, c.src_output, cs.name, iname))
            elif (cs.name, iname) in norm_inputs:
                struct_inputs[iname] = NeuroValue.from_python(
                    norm_inputs[(cs.name, iname)])
            else:
                raise StructureInputError(
                    f"STRUCTURE '{cs.name}': missing required input '{iname}'")
        return struct_inputs

    def _run_structure(self, cs: CompiledStructure,
                       struct_inputs: Dict[str, NeuroValue]
                       ) -> Tuple[Dict[str, NeuroValue], str, Any, float]:
        # cache-key discipline (refinement 2): key accounts for structure
        # identity, definition, inputs, and settings. Structures containing a
        # CONNECT (memory write) are never served from cache: the write must
        # execute every time.
        cache_state = "disabled"
        cache_key = None
        if self.settings.cache_enabled:
            if cs.has_memory_write:
                cache_state = "bypass"
            else:
                fp = "|".join(f"{k}={_fingerprint_value(v)}"
                              for k, v in sorted(struct_inputs.items()))
                cache_key = hashlib.sha256(
                    f"{cs.name}|{cs.def_hash}|{fp}|"
                    f"{self.settings.fingerprint()}".encode("utf-8")).hexdigest()
                if cache_key in self._cache:
                    cached = {k: copy.deepcopy(v)
                              for k, v in self._cache[cache_key].items()}
                    return cached, "hit", None, 0.0
                cache_state = "miss"

        rt = Runtime(cs.graph, max_steps=self.settings.max_steps,
                     timeout_s=self.settings.timeout_s)
        t0 = time.perf_counter()
        try:
            rt.execute(dict(struct_inputs))
        except Exception as e:
            raise StructureExecutionError(
                f"STRUCTURE '{cs.name}' failed: {e}") from e

        out_vals: Dict[str, NeuroValue] = {}
        for oname, nid in cs.outputs.items():
            nv = rt.values.get(nid)
            if nv is None or not isinstance(nv, NeuroValue):
                raise StructureExecutionError(
                    f"STRUCTURE '{cs.name}': output '{oname}' produced no value")
            want = cs.output_dtypes[oname]
            if not _runtime_compatible(nv.dtype, want):
                raise StructureExecutionError(
                    f"STRUCTURE '{cs.name}': output '{oname}' produced "
                    f"{nv.dtype.name}, interface declares {want.name}")
            out_vals[oname] = nv

        if cache_key is not None:
            self._cache[cache_key] = {k: copy.deepcopy(v)
                                      for k, v in out_vals.items()}
        duration = time.perf_counter() - t0
        return out_vals, cache_state, rt.get_memory_dump(), duration
