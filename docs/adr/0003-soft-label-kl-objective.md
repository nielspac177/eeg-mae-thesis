# ADR 0003 — Train on soft labels with KL divergence

- **Status:** Accepted
- **Date:** 2026-06-24

## Context

Each HMS window is annotated by several experts; the label is a **vote distribution** over six
classes, not a single class. The competition scores **KL divergence** between predicted and true
distributions. The original `03b` linear probe instead used hard `argmax` labels with
`LogisticRegression` — discarding the disagreement signal and optimising a different objective
than the one being measured.

## Decision

The supervised stage trains directly on the normalised soft labels with `nn.KLDivLoss`
(wrapped in `SoftLabelKLLoss`, which applies `log_softmax` internally). The probe is a
configurable `MLPHead`; `depth=1` reproduces the logistic-regression baseline for comparison,
`depth=2` is the default 3-layer MLP. Evaluation uses the same KL metric, out-of-fold, with
patient-grouped splits.

## Consequences

- Training and evaluation optimise the same quantity — no train/metric mismatch.
- Soft labels act as a built-in regulariser (label smoothing from real annotator uncertainty),
  which helps on ambiguous epileptiform patterns.
- Experiment 1 is therefore not a separate run but a property of every supervised config; the
  hard-label baseline survives only as the `depth1_linear` variant in experiment 2.
- mixup composes naturally because it convex-combines soft-label targets.
