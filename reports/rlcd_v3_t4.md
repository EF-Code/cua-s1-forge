# RLCD-v3 smoke evaluation

Date: 2026-09-22

## Objective

RLCD-v3 replaces the sampled categorical policy-gradient term used in the v1 smoke run with a deterministic exact expected-reward term. It keeps natural-distribution cross-entropy as the primary loss, applies a small sqrt-inverse-frequency correction, and selects checkpoints by validation NLL with a balanced-macro secondary score.

## Untouched smoke-test result

| Metric | Baseline | RLCD-v3 best-NLL | Change |
|---|---:|---:|---:|
| top-1 | 51.6749% | 52.3339% | +0.659 pp |
| NLL | 1.6714 | 1.6191 | -0.0523 |
| ECE | 0.0355 | 0.0429 | +0.0074 |
| macro action accuracy | 47.9940% | 48.7460% | +0.752 pp |
| fill accuracy | 1.2563% | 4.3970% | 3.50x relative |
| click accuracy | 0.0000% | 0.0000% | unresolved |

The checkpoint was trained for four epochs from the baseline checkpoint with seed `20260924`, learning rate `1e-4`, batch size `128`, and expected-reward coefficient `0.10`.

## Interpretation and limits

This is a validated correction of the v1 regression, not a 100x claim. The run uses one small synthetic smoke split and one seed. The click class remains unsolved, and ECE is slightly worse than baseline. A larger improvement requires more diverse data, hard negatives for click/submit behavior, multiple seeds, and a held-out evaluation suite.
