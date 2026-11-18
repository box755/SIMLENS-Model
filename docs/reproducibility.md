# Reproducibility and configuration disclosure

## Release scope

This package is based on the retained V3 research implementation and the
three full-model best checkpoints, not the earlier LLaVA-NeXT prototype.
The package, classes and entry points use the SIMLENS name throughout.
Shared utilities were moved inside the package;
machine-specific cache and workspace paths were removed. The model forward,
temporal patching, audio pooling, readout, losses and metric definitions are not
redesigned. Additions are packaging, explicit checkpoint validation, input/output
interfaces, dependency pinning and configuration checks.

## Processor grouping is not positional grouping

The inspected implementation groups input tokens in one-second processor chunks.
The cached Qwen Thinker configuration separately uses `seconds_per_chunk=2` and
`position_id_per_seconds=25` (40 ms units). Position triples are computed before
audio pooling. Pooling averages audio embeddings but copies the middle selected
token's triple, and each suffix query copies its group's last visual token triple.
The one-second/two-second mismatch can assign position triples in an order that
differs from the content sequence. The public default deliberately does not repair
that behavior while presenting the original checkpoint results.

The public YAML pins the **inspected cached** backbone revision
`ae9e1690543ffd5c0221dc27f79834d0294cba00`, processor grouping 1 s, positional
grouping 2 s, and 25 IDs/s. The loader checks those position settings rather than
silently accepting a changed upstream default. `sync_position_chunk` stays false.
The historical 54 resolved YAMLs establish processor grouping but do not snapshot
the loaded Thinker config or exact Transformers source. Cache timestamps and the
current source are not run-linked runtime proof. Matching the exact environment
of every historical training run remains unresolved. Assessing a corrected
alignment requires matched retraining/evaluation, not just flipping an inference
flag and retaining the published scores.

## Selection and OOM handling

Training selects checkpoints by validation per-video Spearman, checking every
500 optimizer steps with a minimum improvement of 0.005 and patience 3. Validation
takes the first 150 available videos in official split order, without a fresh
random subsample per evaluation. Historical metrics record 150 videos, but complete
per-run ID manifests were not retained.

The retained code handles training OOM by clearing accumulated gradients and
skipping the failed example; evaluation can skip failed examples. Public evaluation
prints skip events. Record the emitted IDs and count, and do not assume that an OOM
run still used the advertised test subset. The historical saved full-test prediction
files share the 1,874-ID subset, but complete training-time OOM counts are unavailable.

## Scales, ties and scoring

The model head is linear and returns unrestricted scores. Labels are normalized
within each video inside the loss. Listwise supervision uses Gaussian jitter of
1e-6 to resolve target ties. Evaluation retains the original NumPy tie behavior:
top/bottom-15% F1 uses the existing argsort, whereas strict valley precision uses
stable sorting. No fresh random tie-breaking is introduced at test time.

`simlens-evaluate` saves predictions at four decimal places as in the retained
evaluation code. `simlens-score` is the unified paper scorer on saved predictions;
use it for headline mAP, strict vP@k, and RIR. The native evaluation's `valley_hit@k`
is a different metric, not strict vP@k. Small discrepancies can arise between
native pre-serialization metrics and rescoring rounded predictions. The general
`simlens-predict` command preserves float32 output precision rather than silently
rounding it to the benchmark dump format.

Checkpoint `meta.json` records **validation selection** metrics, not test scores.
The published three-seed mean/std summarizes independent models, not averaged
prediction curves. YT-Fresh is evaluated with all three checkpoints: per-video
Spearman is 0.431, 0.439 and 0.436 for seeds 42, 7 and 123 (mean 0.435). Scoring
saved predictions against `datasets/ytfresh/most_replayed.jsonl` gives these values.
