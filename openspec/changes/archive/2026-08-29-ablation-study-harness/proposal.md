## Why

This branch (`ttsgan-direct`) has accumulated three architectural additions on top of the base TTS-GAN — the `MultiscaleDWTDiscriminator` + `StackingDiscriminator` pair, the feature-matching loss (`lambda_fm`), and the `PostNet` residual smoothing (`use_postnet`) — each justified by hypothesis, not measurement. For a thesis, each needs quantitative evidence that it actually improves synthetic EEG quality, not just a plausible mechanism. There is currently no repeatable way to train the relevant configs and compare them on the same metrics.

## What Changes

- Add `ablation_pipeline.py`: a driver script (same style as `moabb_pipeline.py`) that trains a fixed matrix of GAN configs for a chosen subject, generates samples from each resulting checkpoint, and computes a shared metric set per config.
- Refactor the metric functions already in `compare_samples.py` (`spectral_jsd`, `mmd_rbf`, PSD computation, ERP peak amplitude/latency) out into a small importable module (`eeg_metrics.py` or similar) so both `compare_samples.py` and the new ablation script call the same code — no duplicated metric logic.
- Ablation matrix (9 configs, see design.md for exact kwargs): no-DWT/no-PostNet baseline, PostNet-only (no DWT), +DWT (no stacking), +DWT+stacking (isolates the stacking meta-head), and a `lambda_fm ∈ {0, 20, 50}` × `use_postnet ∈ {False, True}` factorial (DWT on, no stacking) — the factorial's `lambda_fm=20, postnet=True` point is the full model.
- Output: one CSV row per config (JSD, MMD², ERP peak amplitude/latency error) plus a multi-config PSD-overlay figure, for direct inclusion in the thesis.

## Capabilities

### New Capabilities
- `ablation-study-harness`: trains a defined config matrix and produces a comparison table + figure quantifying each architectural addition's effect.

### Modified Capabilities
- none (no existing spec's requirements change; `compare_samples.py`'s behavior/output is preserved, only its metric code is relocated)

## Impact

- New file: `ablation_pipeline.py`.
- New file: shared metrics module extracted from `compare_samples.py` (exact name/location decided in design.md).
- Modified: `compare_samples.py` imports metrics from the new module instead of defining them inline (no behavior change).
- Uses existing `train_gan()` (from `moabb_pipeline.py`) and `generate_samples_main.py` unchanged.
- Runs several real GPU training jobs (5 configs × chosen epoch budget) — this is a compute cost, not a code-complexity cost.
- Explicitly out of scope: comparing against the autoencoder-based `main` branch ("vanilla EEG-GAN") or the upstream `tts-gan` reference repo. Different branches/repos, incompatible checkpoint formats. Left as future work — could be added later as a manual "drop in an externally-generated samples CSV" input to the same comparison table, not as cross-repo training automation.
