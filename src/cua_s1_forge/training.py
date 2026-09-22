"""Training helpers layered on top of the upstream CUA-S1 model and collator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
import time
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from .rlcd import RLCDConfig, rlcd_loss


@dataclass(frozen=True)
class RLCDTrainConfig:
    epochs: int = 2
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    seed: int = 7
    grad_clip: float = 1.0

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip positive")


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_verifier_policy(
    model: torch.nn.Module,
    collator: Callable[..., dict[str, torch.Tensor]],
    train_dataset: Any,
    validation_fn: Callable[..., dict[str, Any]] | None = None,
    validation_dataset: Any | None = None,
    device: torch.device | str = "cpu",
    objective: RLCDConfig | None = None,
    config: RLCDTrainConfig | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fine-tune a CUA-S1 model using local verifier rewards.

    A frozen copy of the starting model supplies the KL reference. If a validation
    callback is supplied, the best validation-NLL state is restored before return.
    """

    objective = objective or RLCDConfig()
    config = config or RLCDTrainConfig()
    objective.validate()
    config.validate()
    device = torch.device(device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model.to(device)
    reference = deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] | None = None
    best_nll = float("inf")
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for batch in loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            with torch.no_grad():
                reference_logits = reference(batch)
            loss, details = rlcd_loss(
                logits,
                batch["labels"],
                reference_logits=reference_logits,
                config=objective,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite RLCD loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_size = int(batch["labels"].shape[0])
            examples += batch_size
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size

        train_metrics = {key: value / max(examples, 1) for key, value in totals.items()}
        record: dict[str, Any] = {"epoch": epoch, "examples": examples, "train": train_metrics}
        if validation_fn is not None and validation_dataset is not None:
            validation = validation_fn(model, validation_dataset, collator, device)
            record["validation"] = validation
            validation_nll = float(validation.get("nll", float("inf")))
            if validation_nll < best_nll:
                best_nll = validation_nll
                best_state = _clone_state(model)
                best_validation = dict(validation)
        history.append(record)
        log(f"epoch={epoch} train={train_metrics} validation={record.get('validation')}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_validation": best_validation,
        "best_state": best_state,
        "train_seconds": time.perf_counter() - started,
    }
