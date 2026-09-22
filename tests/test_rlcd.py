import pytest
import torch

from cua_s1_forge.rlcd import RLCDConfig, rlcd_loss, verifier_rewards


def test_verifier_rewards():
    actions = torch.tensor([0, 1, 2])
    labels = torch.tensor([0, 0, 2])
    assert torch.equal(verifier_rewards(actions, labels), torch.tensor([1.0, -1.0, 1.0]))


def test_rlcd_loss_is_finite_and_reports_metrics():
    torch.manual_seed(4)
    logits = torch.randn(8, 4, requires_grad=True)
    reference = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss, metrics = rlcd_loss(
        logits,
        labels,
        reference_logits=reference,
        config=RLCDConfig(),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for key in ("loss", "policy_loss", "ce_loss", "entropy", "kl", "reward_mean", "sample_accuracy"):
        assert key in metrics


def test_invalid_labels_raise():
    with pytest.raises(ValueError):
        rlcd_loss(torch.zeros(2, 3), torch.tensor([0, 4]))
