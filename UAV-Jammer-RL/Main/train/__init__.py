"""Training entrypoints package for UAV-Jammer-RL.

Use modules under ``Main.train.*`` to launch training jobs, e.g.::

    python -m Main.train.train_qmix
    python -m Main.train.train_vdn

For programmatic use, the primary exported callables are listed in ``__all__`` and
loaded lazily via ``__getattr__`` so ``python -m Main.train.<module>`` does not
pre-import the target submodule.
"""

from importlib import import_module

_EXPORTS = {
    "train_mpdqn_iql": ("Main.train.train_iql", "train_mpdqn_iql"),
    "train_mappo": ("Main.train.train_mappo", "train_mappo"),
    "train_mpdqn_qmix": ("Main.train.train_qmix", "train_mpdqn_qmix"),
    "train_qmix_value_expansion": ("Main.train.train_qmix_value_expansion", "train_qmix_value_expansion"),
    "train_qmix_value_expansion_fixed_wm": (
        "Main.train.train_qmix_value_expansion_fixed_wm",
        "train_qmix_value_expansion_fixed_wm",
    ),
    "train_qmix_wm_alternating": ("Main.train.train_qmix_wm_alternating", "train_qmix_wm_alternating"),
    "train_mpdqn_qplex": ("Main.train.train_qplex", "train_mpdqn_qplex"),
    "train_mpdqn_vdn": ("Main.train.train_vdn", "train_mpdqn_vdn"),
    "train_world_model": ("Main.train.train_world_model", "train_world_model"),
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
