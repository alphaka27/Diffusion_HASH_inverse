"""
Notebook-safe progress helpers.
"""

from __future__ import annotations

import os
import sys
import time
from types import TracebackType
from typing import Any, Generic, Iterable, Iterator, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def _is_notebook_kernel() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "JPY_PARENT_PID" in os.environ


class PlainProgress(Generic[T]):
    """
    Minimal progress iterator that avoids Jupyter widget/display state.
    """

    def __init__(
        self,
        iterable: Iterable[T],
        *,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: Optional[str] = None,
        mininterval: float = 5.0,
        disable: bool = False,
        **_: Any,
    ) -> None:
        self.iterable = iterable
        self.total = total if total is not None else self._infer_total(iterable)
        self.desc = desc or "Progress"
        self.unit = unit or "it"
        self.mininterval = mininterval
        self.disable = disable
        self.n = 0
        self._closed = False
        self._last_print = 0.0
        self._postfix: dict[str, Any] = {}

    @staticmethod
    def _infer_total(iterable: Iterable[T]) -> Optional[int]:
        try:
            return len(iterable)  # type: ignore[arg-type]
        except TypeError:
            return None

    def __enter__(self) -> "PlainProgress[T]":
        self._print(force=True)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[T]:
        if self._last_print == 0.0:
            self._print(force=True)
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, n: int = 1) -> None:
        self.n += n
        self._print()

    def set_postfix(self, ordered_dict: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        self._postfix = {}
        if ordered_dict:
            self._postfix.update(ordered_dict)
        self._postfix.update(kwargs)
        self._print()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._print(force=True, closing=True)
        sys.stdout.flush()

    def _print(self, *, force: bool = False, closing: bool = False) -> None:
        if self.disable:
            return
        now = time.monotonic()
        reached_total = self.total is not None and self.n >= self.total
        if not force and not reached_total and now - self._last_print < self.mininterval:
            return

        self._last_print = now
        total = "?" if self.total is None else str(self.total)
        status = "done " if closing else ""
        postfix = ""
        if self._postfix:
            postfix = " " + " ".join(f"{key}={value}" for key, value in self._postfix.items())
        print(f"{self.desc}: {status}{self.n}/{total} {self.unit}{postfix}", flush=True)


def progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: Optional[str] = None,
    mininterval: float = 5.0,
    **kwargs: Any,
) -> tqdm[T] | PlainProgress[T]:
    """
    Return tqdm in terminal contexts and plain line progress in notebooks.
    """
    if _is_notebook_kernel():
        return PlainProgress(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            mininterval=mininterval,
            **kwargs,
        )

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        mininterval=mininterval,
        **kwargs,
    )
