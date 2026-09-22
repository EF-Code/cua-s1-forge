from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ExactExpectedRewardConfig:
    """Conservative objective for deterministic reward-model fine-tuning.

    The supervised cross-entropy anchor remains primary. The reward term uses
    the model's exact probability of the gold action, avoiding sampled-action
    variance and preserving a probability-aware training signal.
    """

    supervised_coef: float = 1.0
    expected_reward_coef: float = 0.10
    action_weight_power: float = 0.50

    def validate(self) -> None:
        if self.supervised_coef <= 0:
            raise ValueError("supervised_coef must be positive")
        if self.expected_reward_coef < 0:
            raise ValueError("expected_reward_coef must be non-negative")
        if self.action_weight_power < 0:
            raise ValueError("action_weight_power must be non-negative")


def inverse_frequency_action_weights(
    actions: Sequence[str], power: float = 0.50
) -> dict[str, float]:
    """Return sqrt-inverse-frequency weights, normalized only by usage."""
    if not actions:
        raise ValueError("actions must not be empty")
    if power < 0:
        raise ValueError("power must be non-negative")
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    total = float(len(actions))
    return {action: (total / count) ** power for action, count in counts.items()}


def exact_expected_reward_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
    """Negative exact expected reward for a one-hot correct-action reward.

    ``action_weights`` maps class indices to positive weights. Unlike a sampled
    policy-gradient estimate, this is deterministic for a given minibatch and
    directly rewards probability assigned to the correct action.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.numel() and (labels.min() < 0 or labels.max() >= logits.shape[1]):
        raise ValueError("labels contain an invalid class index")

    labels = labels.to(dtype=torch.long, device=logits.device)
    probabilities = logits.softmax(dim=-1)
    gold_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    if action_weights is None:
        return -gold_probability.mean()

    class_weights = torch.ones(logits.shape[1], device=logits.device, dtype=logits.dtype)
    for index, weight in action_weights.items():
        if index < 0 or index >= logits.shape[1]:
            raise ValueError("action_weights contains an invalid class index")
        if weight <= 0:
            raise ValueError("action weights must be positive")
        class_weights[index] = float(weight)
    selected_weights = class_weights[labels]
    return -(gold_probability * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)


def conservative_expected_reward_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_weights: Mapping[int, float] | None = None,
    config: ExactExpectedRewardConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Cross-entropy plus a small exact expected-reward correction."""
    config = config or ExactExpectedRewardConfig()
    config.validate()
    ce = F.cross_entropy(logits, labels)
    expected_reward = exact_expected_reward_loss(logits, labels, action_weights)
    return {
        "loss": config.supervised_coef * ce + config.expected_reward_coef * expected_reward,
        "ce": ce,
        "expected_reward_loss": expected_reward,
    }
