"""Model-agnostic progress reporting for every loop in the training pipeline.

All student models drive the same helper, so switching the experiment between
``mlp``, ``resnet50_1d``, ``lstm`` and the tabular ``xgboost`` path keeps the
output format identical. Interactive runs get a tqdm bar; runs whose output is
redirected (nohup, tee, CI) get one greppable line every ``log_every`` items.
"""

from __future__ import annotations

import sys
import time
from typing import Any

_MODES = ("auto", "bar", "log", "off")


def _load_tqdm() -> Any:
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return None


def _is_tty(stream: Any) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def resolve_mode(mode: str, stream: Any = None) -> str:
    """Pick the effective mode, degrading gracefully when tqdm is unavailable."""
    if mode not in _MODES:
        raise ValueError(f"Unknown progress mode {mode!r}; choose one of: {', '.join(_MODES)}")
    if mode in ("log", "off"):
        return mode
    available = _load_tqdm() is not None
    if mode == "bar":
        return "bar" if available else "log"
    target = sys.stderr if stream is None else stream
    return "bar" if available and _is_tty(target) else "log"


class BatchProgress:
    """Progress for one loop, usable as a context manager."""

    def __init__(self, desc: str, *, total: int | None = None, mode: str = "auto",
                 log_every: int = 100, unit: str = "batch", stream: Any = None):
        self.desc = desc
        self.total = total
        self.unit = unit
        self.log_every = max(int(log_every), 0)
        if self.total and self.log_every > 0:
            # Keep short loops observable instead of printing only the final line.
            self.log_every = min(self.log_every, max(1, self.total // 10))
        self.mode = resolve_mode(mode, stream)
        self._started = time.perf_counter()
        self._bar: Any = None
        if self.mode == "bar":
            self._bar = _load_tqdm()(total=total, desc=desc, unit=unit, file=sys.stderr,
                                     dynamic_ncols=True, leave=False)

    def update(self, index: int, *, average_loss: float | None = None) -> None:
        if self._bar is not None:
            self._bar.update(1)
            if average_loss is not None:
                self._bar.set_postfix(loss=f"{average_loss:.4f}")
            return
        if self.mode != "log" or self.log_every <= 0:
            return
        finished = self.total is not None and index == self.total
        if index % self.log_every != 0 and not finished:
            return
        elapsed = time.perf_counter() - self._started
        rate = index / max(elapsed, 1e-6)
        eta = (self.total - index) / rate if self.total else float("nan")
        suffix = "" if average_loss is None else f" loss_avg={average_loss:.4f}"
        print(
            f"{self.desc} progress={index}/{self.total or '?'} rate={rate:.2f} batches_per_sec "
            f"eta_sec={eta:.1f}{suffix}",
            flush=True,
        )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "BatchProgress":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def make_progress(desc: str | None, *, total: int | None = None,
                  options: dict[str, Any] | None = None) -> BatchProgress:
    """Build a tracker for ``desc``; passing ``None`` disables the output."""
    settings = dict(options or {})
    if desc is None:
        settings["mode"] = "off"
        desc = "batches"
    return BatchProgress(desc, total=total, **settings)