"""Evaluation entrypoints package for UAV-Jammer-RL.

Use modules under ``Main.evaluate.*`` to launch evaluation jobs, e.g.::

    python -m Main.evaluate.evaluate_all_baselines --episodes 100 --steps 1000
    python -m Main.evaluate.evaluate_mpdqn --mode mpdqn --weights <weights.pth>
    python -m Main.evaluate.evaluate_mappo --weights <mappo_weights.pth>
    python -m Main.evaluate.run_heuristic --policy greedy_sensing

For programmatic use, the primary exported callables are listed in ``__all__`` and
loaded lazily via ``__getattr__`` so ``python -m Main.evaluate.<module>`` does not
pre-import the target submodule.
"""

from importlib import import_module

_EXPORTS = {
    "evaluate_all_baselines": ("Main.evaluate.evaluate_all_baselines", "evaluate_all_baselines"),
    "build_all_eval_arg_parser": ("Main.evaluate.evaluate_all_baselines", "build_arg_parser"),
    "evaluate_all_baselines_main": ("Main.evaluate.evaluate_all_baselines", "main"),
    "evaluate_policy": ("Main.evaluate.evaluate_mpdqn", "evaluate_policy"),
    "build_mpdqn_eval_arg_parser": ("Main.evaluate.evaluate_mpdqn", "build_arg_parser"),
    "evaluate_mpdqn_main": ("Main.evaluate.evaluate_mpdqn", "main"),
    "evaluate_mappo": ("Main.evaluate.evaluate_mappo", "evaluate_mappo"),
    "build_mappo_eval_arg_parser": ("Main.evaluate.evaluate_mappo", "build_arg_parser"),
    "evaluate_mappo_main": ("Main.evaluate.evaluate_mappo", "main"),
    "run_heuristic": ("Main.evaluate.run_heuristic", "run_heuristic"),
    "build_heuristic_arg_parser": ("Main.evaluate.run_heuristic", "build_arg_parser"),
    "run_heuristic_main": ("Main.evaluate.run_heuristic", "main"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
