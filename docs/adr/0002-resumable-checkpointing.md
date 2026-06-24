# ADR 0002 — Per-epoch resumable checkpointing

- **Status:** Accepted
- **Date:** 2026-06-24

## Context

Training runs on a single Apple-silicon Mac (MPS), where a full 90-epoch MAE pretrain plus the
classification sweeps take many hours. Runs must survive interruptions — closing the laptop, a
crash, or a deliberate stop — without losing progress or corrupting state. The hardware is fixed
(no cloud GPU), so the design has to make long, interruptible local runs safe.

## Decision

A single `ResumableTrainer` owns the loop. Every epoch it writes one checkpoint
(`ckpt_last.pt`) containing the model, optimizer, scheduler, loss history, and **all RNG states**
(the trainer's own DataLoader generator plus the global torch/numpy/MPS generators), via an
atomic temp-then-replace write. On construction it auto-loads any existing checkpoint and sets
`start_epoch` to continue. OOF studies checkpoint per fold; completed studies are cached to
`results/oof/<name>.npz` and skipped on re-run (idempotent).

We assert a **resume-equivalence** property, tested on CPU: training N epochs straight equals
training k epochs, reconstructing the trainer from the checkpoint, and training the remaining
N−k — bit-for-bit.

## Consequences

- Any run can be killed and resumed by re-issuing the same command; nothing recomputes.
- Bit-exact resume is guaranteed only on CPU (where all RNG is serialisable). On MPS the Metal
  RNG is captured best-effort; resume is always *correct* (continues from saved weights and
  optimizer) but not guaranteed identical to an uninterrupted run. This is an accepted trade-off:
  scientific conclusions rest on converged metrics, not on a specific RNG trajectory.
- A small storage cost (one checkpoint per run/fold) and a one-write-per-epoch overhead.
