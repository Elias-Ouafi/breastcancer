# 0007 — GroupNorm from scratch over an ImageNet-pretrained encoder

**Status:** accepted · **In effect:** 2026-08-02 · **Journal:** §4.1

## Context

The literature recommends pretrained encoders for small datasets, and the code supported
one (`--architecture pretrained`, a ResNet-34 U-Net). Measured on 186 patients, 30
epochs, identical configuration otherwise:

| Configuration | Test Dice | Validation Dice |
|---|:---:|---|
| Pretrained encoder (46 BatchNorm layers) | 0.414 | collapsed to 0.000 from epoch 10 |
| From scratch, GroupNorm (18 layers, 0 BatchNorm) | **0.550** | peak 0.655 |

The cause: with a tiny lesion fraction and small batches, BatchNorm running statistics
never converge to inference conditions.

## Decision

Keep the from-scratch GroupNorm U-Net. **Delete** the pretrained option, its arguments
and the `segmentation-models-pytorch` dependency, rather than leave a flag that silently
produces a collapsed model. The slice and exam classifiers reuse the same GroupNorm
blocks for the same reason.

## Consequences

- One fewer dependency, and no silent failure mode behind a CLI flag.
- The only other gain in that round was engineering, not accuracy: the memory-mapped
  slice bank cut an epoch from 1,020 s to 143 s (7.1×).
- Untested lead, kept in the journal: convert BatchNorm to GroupNorm while keeping the
  pretrained convolution weights.
