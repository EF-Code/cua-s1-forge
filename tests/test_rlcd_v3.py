import pytest
import torch

from cua_s1_forge.rlcd_v3 import (
    ExactExpectedRewardConfig,
    conservative_expected_reward_loss,
    exact_expected_reward_loss,
    inverse_frequency_action_weights,
)


def test_inverse_frequency_weights_prioritize_rare_actions():
    weights = inverse_frequency_action_weights(["skip"] * 8 + ["fill"] * 2)
    assert weights["fill"] > weights["skip"]


def test_expected_reward_loss_is_finite_and_differentiable():
    logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = exact_expected_reward_loss(logits, labels, {0: 1.0, 1: 2.0})
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_conservative_loss_has_supervised_anchor():
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 1])
    result = conservative_expected_reward_loss(
        logits,
        labels,
        config=ExactExpectedRewardConfig(expected_reward_coef=0.10),
    )
    assert set(result) == {"loss", "ce", "expected_reward_loss"}
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_invalid_shapes_fail_closed():
    with pytest.raises(ValueError):
        exact_expected_reward_loss(torch.randn(2, 3, 1), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        exact_expected_reward_loss(torch.randn(2, 3), torch.tensor([[0, 1]]))
