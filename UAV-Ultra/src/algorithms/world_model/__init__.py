"""世界模型组件层（plan locked #3：**无独立训练器**）。

提供给 Stage 5 ``wm_alternating`` callback 与 QMIX value-expansion 训练循环消费：
- ``JointWorldModel`` / ``JointWorldModelConfig`` / ``RSSMHiddenState`` / ``RSSMObserveOutput``
- ``encode_joint_action_exec`` / ``exec_action_dim``
- ``imagine_rollout``（模型想象 rollout）
- ``TDlambdaConfig`` / ``rollout_td_lambda_return`` / ``td_lambda_truncated``
- ``compute_wm_losses`` 等 loss 原子
- ``WorldModelSequenceReplayBuffer``
"""

from src.algorithms.world_model.action_encoding import (
    encode_agent_action_exec,
    encode_joint_action_exec,
    exec_action_dim,
)
from src.algorithms.world_model.losses import (
    compute_wm_losses,
    kl_loss,
    reward_loss,
    state_delta_loss,
)
from src.algorithms.world_model.model import (
    JointWorldModel,
    JointWorldModelConfig,
    RSSMHiddenState,
    RSSMObserveOutput,
)
from src.algorithms.world_model.replay_buffer import WorldModelSequenceReplayBuffer
from src.algorithms.world_model.rollout import imagine_rollout
from src.algorithms.world_model.value_expansion import (
    TDlambdaConfig,
    rollout_td_lambda_return,
    td_lambda_truncated,
)

__all__ = [
    # model
    "JointWorldModel", "JointWorldModelConfig", "RSSMHiddenState", "RSSMObserveOutput",
    # action encoding
    "exec_action_dim", "encode_agent_action_exec", "encode_joint_action_exec",
    # rollout / value-expansion
    "imagine_rollout", "TDlambdaConfig", "rollout_td_lambda_return", "td_lambda_truncated",
    # losses
    "state_delta_loss", "reward_loss", "kl_loss", "compute_wm_losses",
    # replay
    "WorldModelSequenceReplayBuffer",
]
