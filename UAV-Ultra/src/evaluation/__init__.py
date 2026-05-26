"""Stage 6 evaluation entry points."""

from __future__ import annotations

__all__ = ["EvaluationResult", "run_evaluation"]


def __getattr__(name: str):
    if name in {"EvaluationResult", "run_evaluation"}:
        from src.evaluation import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
