"""Stage 5 training entry points."""

from __future__ import annotations

__all__ = ["TrainingResult", "SubprocVecEnv", "make_fixed_p_trans", "run_training"]


def __getattr__(name: str):
    if name in {"TrainingResult", "run_training"}:
        from src.training import runner

        return getattr(runner, name)
    if name in {"SubprocVecEnv", "make_fixed_p_trans"}:
        from src.training import vec_env

        return getattr(vec_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
