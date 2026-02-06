"""
Decorators for tracing local variable changes inside instance methods.

This module is designed to keep algorithm code unchanged:
- Apply decorators via runtime monkey-patching, or by annotating methods.
- Uses `sys.monitoring` (Python 3.12+, PEP 669) when available to avoid `sys.settrace`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import CodeType
from typing import Any, Callable, Mapping, MutableMapping, Optional

import copy
import sys
import threading
from functools import wraps


def _ordinal(n: int) -> str:
    assert n > 0, "n must be positive"
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _key(n: int, label: str) -> str:
    return f"{_ordinal(n)} {label}"


def _get_logs_dict(logs: Any) -> Optional[MutableMapping[str, Any]]:
    if isinstance(logs, dict):
        return logs
    inner = getattr(logs, "logs", None)
    if isinstance(inner, dict):
        return inner
    return None


@dataclass(frozen=True)
class IndexSpec:
    """
    Resolve an integer index from a local variable name.

    Examples:
        IndexSpec(label="Loop", var="i", transform=lambda x: x + 1)
        IndexSpec(label="Round", var="round_idx")
    """

    label: str
    var: str
    transform: Optional[Callable[[int], int]] = None

    def resolve(self, locals_: Mapping[str, Any]) -> Optional[int]:
        raw = locals_.get(self.var)
        if not isinstance(raw, int):
            return None
        value = self.transform(raw) if self.transform is not None else raw
        if not isinstance(value, int) or value <= 0:
            return None
        return value


@dataclass
class _TraceState:
    logs: Any
    step_idx: int
    watch: tuple[str, ...]
    rename: Mapping[str, str]
    round_spec: Optional[IndexSpec]
    loop_spec: IndexSpec
    auto_round: bool
    require_all: bool
    snapshot: Callable[[Any], Any]

    round_label: str
    loop_label: str

    current_round: int = 1
    prev_loop_raw: Optional[int] = None

    active_key: Optional[tuple[int, int]] = None  # (round_idx, loop_idx)
    active_snapshot: Optional[dict[str, Any]] = None

    def _emit(self, key: tuple[int, int], snapshot: Mapping[str, Any]) -> None:
        if not snapshot:
            return
        round_idx, loop_idx = key
        step_key = _key(self.step_idx, "Step")
        round_key = _key(round_idx, self.round_label)
        loop_key = _key(loop_idx, self.loop_label)

        update_step = getattr(self.logs, "update_step", None)
        if callable(update_step):
            try:
                update_step(
                    self.step_idx,
                    dict(snapshot),
                    level2=(self.round_label, round_idx),
                    level3=(self.loop_label, loop_idx),
                )
                return
            except TypeError:
                pass

        set_value = getattr(self.logs, "set_value", None)
        if callable(set_value):
            try:
                set_value((step_key, round_key, loop_key), dict(snapshot))
                return
            except TypeError:
                pass

        logs_dict = _get_logs_dict(self.logs)
        if logs_dict is None:
            return
        step_node = logs_dict.setdefault(step_key, {})
        if not isinstance(step_node, dict):
            step_node = {}
            logs_dict[step_key] = step_node
        round_node = step_node.setdefault(round_key, {})
        if not isinstance(round_node, dict):
            round_node = {}
            step_node[round_key] = round_node
        round_node[loop_key] = dict(snapshot)

    def flush(self) -> None:
        if self.active_key is None or self.active_snapshot is None:
            return
        self._emit(self.active_key, self.active_snapshot)

    def on_locals(self, locals_: Mapping[str, Any]) -> None:
        loop_raw = locals_.get(self.loop_spec.var)
        if not isinstance(loop_raw, int):
            return
        loop_idx = self.loop_spec.resolve(locals_)
        if loop_idx is None:
            return

        if self.round_spec is not None:
            round_idx = self.round_spec.resolve(locals_)
            if round_idx is None:
                return
        elif self.auto_round:
            if self.prev_loop_raw is not None and loop_raw < self.prev_loop_raw:
                self.current_round += 1
            round_idx = self.current_round
        else:
            round_idx = 1

        self.prev_loop_raw = loop_raw

        snapshot: dict[str, Any] = {}
        for name in self.watch:
            if name not in locals_:
                if self.require_all:
                    return
                continue
            key = self.rename.get(name, name)
            snapshot[key] = self.snapshot(locals_[name])

        if not snapshot:
            return

        key = (round_idx, loop_idx)
        if self.active_key is None:
            self.active_key = key
            self.active_snapshot = snapshot
            return

        if key != self.active_key:
            self._emit(self.active_key, self.active_snapshot or {})
            self.active_key = key
            self.active_snapshot = snapshot
            return

        self.active_snapshot = snapshot


_LOCK = threading.RLock()
_MONITORING_READY = False
_MONITORING_TOOL_ID: Optional[int] = None
_ACTIVE: MutableMapping[tuple[int, CodeType], list[_TraceState]] = {}


def _find_target_frame(code: CodeType) -> Optional[Any]:
    try:
        frame = sys._getframe(1)
    except ValueError:
        return None
    if frame.f_code is code:
        return frame
    cur = frame
    while cur is not None and cur.f_code is not code:
        cur = cur.f_back
    return cur


def _get_state(code: CodeType) -> Optional[_TraceState]:
    tid = threading.get_ident()
    with _LOCK:
        stack = _ACTIVE.get((tid, code))
        if not stack:
            return None
        return stack[-1]


def _on_line(code: CodeType, line: int) -> None:  # noqa: ARG001 - required by sys.monitoring
    state = _get_state(code)
    if state is None:
        return
    frame = _find_target_frame(code)
    if frame is None:
        return
    state.on_locals(frame.f_locals)


def _on_return(code: CodeType, offset: int, retval: Any) -> None:  # noqa: ARG001
    state = _get_state(code)
    if state is None:
        return
    frame = _find_target_frame(code)
    if frame is None:
        return
    state.on_locals(frame.f_locals)


def _ensure_monitoring() -> bool:
    global _MONITORING_READY, _MONITORING_TOOL_ID
    if _MONITORING_READY:
        return True
    if not hasattr(sys, "monitoring"):
        return False

    m = sys.monitoring
    tool_id = 4
    try:
        if m.get_tool(tool_id) is None:
            m.use_tool_id(tool_id, "diffusion_hash_inv.local_var_logger")
    except ValueError:
        # Tool already in use (possibly by another component)
        return False

    m.register_callback(tool_id, m.events.LINE, _on_line)
    m.register_callback(tool_id, m.events.PY_RETURN, _on_return)

    _MONITORING_TOOL_ID = tool_id
    _MONITORING_READY = True
    return True


def trace_local_state(
    *,
    step_idx: int,
    watch: tuple[str, ...],
    loop: IndexSpec,
    round_: Optional[IndexSpec] = None,
    rename: Optional[Mapping[str, str]] = None,
    auto_round: bool = True,
    require_all: bool = True,
    snapshot: Optional[Callable[[Any], Any]] = None,
    backend: str = "auto",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator factory to trace local variable states and store them into `self.logs`.

    Intended use:
        - Wrap a method without editing its source:
          MD5.step4 = trace_local_state(...)(MD5.step4)

    Parameters:
        step_idx:
            Step number passed to `self.logs.update_step(step_idx, ...)`.
        watch:
            Local variable names to snapshot.
        loop:
            IndexSpec for loop index (required). Used as level3 (e.g., "Loop").
        round_:
            IndexSpec for round index (optional). Used as level2 (e.g., "Round").
            If omitted and auto_round=True, rounds are inferred by detecting loop index resets.
        rename:
            Optional mapping of local var name -> output key name.
            Example: {"a": "A", "b": "B", "c": "C", "d": "D"}
        auto_round:
            When round_ is None, infer rounds by detecting loop index resets (e.g., 63 -> 0).
        require_all:
            If True, skip logging until all watched locals exist.
        snapshot:
            Function to snapshot values (defaults to copy.deepcopy).
        backend:
            "auto" | "monitoring" | "trace"
            - auto: prefer sys.monitoring when available.
            - monitoring: require sys.monitoring.
            - trace: use sys.settrace fallback.
    """

    assert isinstance(step_idx, int) and step_idx > 0, "step_idx must be a positive int"
    assert watch, "watch must be non-empty"
    assert isinstance(loop, IndexSpec), "loop must be an IndexSpec"

    rename_map = dict(rename or {})
    snapshot_fn = snapshot or copy.deepcopy

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        target_code = func.__code__

        @wraps(func)
        def _run_with_monitoring(self, *args, **kwargs):
            if not _ensure_monitoring():
                if backend == "monitoring":
                    raise RuntimeError("sys.monitoring is unavailable or tool_id is already in use")
                return _run_with_trace(self, *args, **kwargs)

            logs = getattr(self, "logs", None)
            if logs is None:
                return func(self, *args, **kwargs)

            state = _TraceState(
                logs=logs,
                step_idx=step_idx,
                watch=tuple(watch),
                rename=rename_map,
                round_spec=round_,
                loop_spec=loop,
                auto_round=auto_round,
                require_all=require_all,
                snapshot=snapshot_fn,
                round_label=(round_.label if round_ is not None else "Round"),
                loop_label=loop.label,
            )

            tid = threading.get_ident()
            m = sys.monitoring
            tool_id = _MONITORING_TOOL_ID
            assert tool_id is not None

            prev_events = m.get_local_events(tool_id, target_code)
            need = m.events.LINE | m.events.PY_RETURN

            with _LOCK:
                _ACTIVE.setdefault((tid, target_code), []).append(state)
            m.set_local_events(tool_id, target_code, prev_events | need)

            try:
                return func(self, *args, **kwargs)
            finally:
                m.set_local_events(tool_id, target_code, prev_events)
                with _LOCK:
                    stack = _ACTIVE.get((tid, target_code), [])
                    if stack:
                        stack.pop()
                    if not stack:
                        _ACTIVE.pop((tid, target_code), None)
                state.flush()

        @wraps(func)
        def _run_with_trace(self, *args, **kwargs):
            logs = getattr(self, "logs", None)
            if logs is None:
                return func(self, *args, **kwargs)

            state = _TraceState(
                logs=logs,
                step_idx=step_idx,
                watch=tuple(watch),
                rename=rename_map,
                round_spec=round_,
                loop_spec=loop,
                auto_round=auto_round,
                require_all=require_all,
                snapshot=snapshot_fn,
                round_label=(round_.label if round_ is not None else "Round"),
                loop_label=loop.label,
            )

            prev = sys.gettrace()

            def local_tracer(frame, event, arg):
                if event in ("line", "return"):
                    state.on_locals(frame.f_locals)
                if prev is not None:
                    prev(frame, event, arg)
                return local_tracer

            def global_tracer(frame, event, arg):
                if event == "call" and frame.f_code is target_code:
                    return local_tracer
                if prev is not None:
                    return prev(frame, event, arg)
                return None

            sys.settrace(global_tracer)
            try:
                return func(self, *args, **kwargs)
            finally:
                sys.settrace(prev)
                state.flush()

        use_monitoring = backend in ("auto", "monitoring")
        if use_monitoring and hasattr(sys, "monitoring"):
            return _run_with_monitoring
        return _run_with_trace

    return decorator
