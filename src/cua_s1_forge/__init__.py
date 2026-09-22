"""Research extensions for the upstream CUA-S1 scorer."""

from .rlcd import RLCDConfig, rlcd_loss, verifier_rewards
from .training import RLCDTrainConfig, train_verifier_policy

__all__ = [
    "RLCDConfig",
    "RLCDTrainConfig",
    "rlcd_loss",
    "train_verifier_policy",
    "verifier_rewards",
]
