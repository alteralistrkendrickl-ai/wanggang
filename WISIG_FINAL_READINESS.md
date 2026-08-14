# WiSig final-readiness audit

`wisig_final_readiness_audit.py` consolidates the frozen validation evidence for
the simplified paired comparison:

- cross-rx: B0 versus A1-C;
- cross-day: B0 versus A1-S;
- training seeds 2024--2028;
- 1/5/10/15/20-shot, with 100 support-set seeds per checkpoint.

The audit hashes 20 `best_encoder.pth` files and matches each hash to exactly one
500-row validation `iterations.csv`.  It rejects duplicates, missing cells,
incorrect roles, protocols, checkpoint hashes, and seed schedules.  Differences
are paired by support-set seed and summarized with training seed as the
statistical unit.

The frozen seed-2024 B0 artifacts live in the sibling `wanggang_wisig_audit`
project, whereas A1 and the 2025--2028 training-seed artifacts live in
`wanggang_wisig_a1_fair`.  The audit treats these roots explicitly; it does not
silently copy or rename checkpoints.

This command does **not** load final arrays, does **not** run a model, and does
**not** authorize final unsealing.  The current `wisig_final_evaluate.py` remains
stale and must not be run.

After this validation-evidence audit passes, `wisig_final_matrix_prepare.py`
performs the next read-only stage.  It verifies the exact readiness-report hash,
rehashes all 20 checkpoints, and hashes the eight final NPY files as opaque
bytes.  It writes `FROZEN_MANIFEST_DRAFT.json`, while keeping `RUN_ENABLED=False`.
It neither imports NumPy nor contains an array loader.

Expected server runtime: about 10--60 seconds, dominated by hashing 20 encoder
checkpoints.

```bash
python wisig_final_readiness_audit.py \
  --project-root /home/yuanlong/yl/wanggang_wisig_a1_fair \
  --baseline-2024-root /home/yuanlong/yl/wanggang_wisig_audit
```
