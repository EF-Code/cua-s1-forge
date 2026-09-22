"""Verifier-sampled policy-gradient objective for CUA-S1 Forge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RLCDConfig:
    """Controls for the local verifier-driven objective."""

    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    entropy_coef: float = 0.002
    kl_coef: float = 0.02
    supervised_coef: float = 0.15
    temperature: float = 1.0

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        for name in ("entropy_coef", "kl_coef", "supervised_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.reward_correct <= self.reward_wrong:
            raise ValueError("reward_correct must be greater than reward_wrong")


def verifier_rewards(
    actions: torch.Tensor,
    labels: torch.Tensor,
    correct: float = 1.0,
    wrong: float = -1.0,
) -> torch.Tensor:
    """Return verifier rewards for sampled actions against gold labels."""

    if actions.shape != labels.shape:
        raise ValueError("actions and labels must have the same shape")
    correct_value = torch.as_tensor(correct, dtype=torch.float32, device=actions.device)
    wrong_value = torch.as_tensor(wrong, dtype=torch.float32, device=actions.device)
    return torch.where(actions.eq(labels), correct_value, wrong_value)


def _validate_logits_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, choices]")
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= logits.shape[1]):
        raise ValueError("labels contain an invalid choice index")
    return labels


def rlcd_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor | None = None,
    config: RLCDConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a sampled verifier policy loss with optional KL anchoring.

    The verifier reward is exact on the local label: sampled correct actions receive
    reward_correct and all other actions receive reward_wrong. The baseline is the
    model's expected reward, which keeps the policy-gradient estimator centered.
    """

    config = config or RLCDConfig()
    config.validate()
    labels = _validate_logits_labels(logits, labels)
    if reference_logits is not None and reference_logits.shape != logits.shape:
        raise ValueError("reference_logits must have the same shape as logits")

    scaled_logits = logits / config.temperature
    distribution = torch.distributions.Categorical(logits=scaled_logits)
    sampled_actions = distribution.sample()
    sampled_log_prob = distribution.log_prob(sampled_actions)
    probabilities = distribution.probs
    gold_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)

    expected_reward = config.reward_wrong + (
        config.reward_correct - config.reward_wrong
    ) * gold_probability
    sampled_reward = verifier_rewards(
        sampled_actions,
        labels,
        correct=config.reward_correct,
        wrong=config.reward_wrong,
    ).to(dtype=logits.dtype)
    advantage = sampled_reward - expected_reward.detach()
    policy_loss = -(advantage * sampled_log_prob).mean()
    ce_loss = F.cross_entropy(logits, labels)
    entropy = distribution.entropy().mean()

    kl_value = logits.new_zeros(())
    if reference_logits is not None:
        reference_logits = reference_logits.to(device=logits.device, dtype=logits.dtype)
        reference_scaled = reference_logits / config.temperature
        current_log_probs = F.log_softmax(scaled_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_scaled, dim=-1)
        kl_terms = probabilities * (current_log_probs - reference_log_probs)
        kl_terms = torch.where(probabilities > 0, kl_terms, torch.zeros_like(kl_terms))
        kl_value = kl_terms.sum(dim=-1).mean()

    total = (
        policy_loss
        + config.supervised_coef * ce_loss
        - config.entropy_coef * entropy
        + config.kl_coef * kl_value
    )
    metrics: dict[str, float] = {
        "loss": float(total.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "ce_loss": float(ce_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "kl": float(kl_value.detach().cpu()),
        "reward_mean": float(sampled_reward.detach().mean().cpu()),
        "sample_accuracy": float(sampled_actions.eq(labels).float().mean().cpu()),
        "gold_probability": float(gold_probability.detach().mean().cpu()),
    }
    return total, metrics
