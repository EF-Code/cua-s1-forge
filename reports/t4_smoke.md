# T4 smoke report (experimental)

Date: 2026-09-22
Upstream source: trycua/cua at commit 27a318c3a616f9ff19d24fe2acca7517c5f8fa7b
Runtime: Google Colab, NVIDIA Tesla T4, torch 2.11.0+cu128, Python 3.13.15

## Dataset

The upstream deterministic synthetic generator produced 600 episodes with a form-signature split:

- train: 12,079 rows from 467 episodes
- validation: 1,631 rows from 62 episodes
- test: 1,821 rows from 71 episodes

The test split was not used for model selection.

## Runs

The model was the CUA-S1 tiny Transformer configuration: width 128, rank 128, context limit 224, option limit 96, two context layers, four heads, and 706,048 trainable parameters.

| Run | Validation top-1 | Validation NLL | Test top-1 | Test NLL | Test ECE | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| supervised baseline, 3 epochs | 0.53342 | 1.59755 | 0.51675 | 1.67140 | 0.03551 | 148.875 s |
| verifier RLCD, 2 epochs | 0.54506 | 1.77781 | 0.51949 | 1.89641 | 0.19305 | 117.042 s |

RLCD used sampled verifier rewards (+1 correct, -1 wrong), entropy coefficient 0.002, KL coefficient 0.02, supervised coefficient 0.15, temperature 1.0, and learning rate 5e-4. The verifier was the local synthetic label function; no Jev or TypeSafe output was used.

## Interpretation

This smoke run validates that the objective trains on a T4 and saves a loadable checkpoint. It does not establish a quality win. Test top-1 increased by 0.00275, while NLL increased by 0.22501 and ECE increased by 0.15754. Fill accuracy rose from 0.01256 to 0.02889, but check and skip accuracy declined. The next experiment should preserve the supervised anchor more strongly and tune reward/advantage estimation against repeated seeds before any broader claim.

Generated data and checkpoints are ignored by Git. This report contains metrics only; it does not publish weights, credentials, or private teacher outputs.
