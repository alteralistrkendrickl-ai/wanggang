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

This command does **not** load final arrays, does **not** run a model, and does
**not** authorize final unsealing.  The current `wisig_final_evaluate.py` remains
stale and must not be run.

Expected server runtime: about 10--60 seconds, dominated by hashing 20 encoder
checkpoints.

```bash
python wisig_final_readiness_audit.py
```
