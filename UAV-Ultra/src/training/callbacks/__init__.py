"""Stage 5+ training callbacks."""

from src.training.callbacks.base import (
    ALLOWED_CALLBACKS,
    CANONICAL_CALLBACK_ORDER,
    CallbackManager,
    TrainHookContext,
    TrainingCallback,
    build_callbacks,
    canonicalize_callback_names,
)
from src.training.callbacks.critic_stable import CriticStableCallback
from src.training.callbacks.jammer_prediction import JammerPredictionCallback
from src.training.callbacks.policy_mobility import PolicyMobilityCallback
from src.training.callbacks.value_expansion import ValueExpansionCallback
from src.training.callbacks.wm_block_alternating import WMBlockAlternatingCallback
from src.training.callbacks.wm_concurrent import WMConcurrentCallback

# Stage 7: deprecated alias; importing the shim module emits FutureWarning.
WMAlternatingCallback = WMConcurrentCallback

__all__ = [
    "ALLOWED_CALLBACKS",
    "CANONICAL_CALLBACK_ORDER",
    "CallbackManager",
    "CriticStableCallback",
    "JammerPredictionCallback",
    "PolicyMobilityCallback",
    "TrainHookContext",
    "TrainingCallback",
    "ValueExpansionCallback",
    "WMAlternatingCallback",
    "WMBlockAlternatingCallback",
    "WMConcurrentCallback",
    "build_callbacks",
    "canonicalize_callback_names",
]
