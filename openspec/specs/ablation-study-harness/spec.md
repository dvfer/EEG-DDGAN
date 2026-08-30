## Purpose

Trains and evaluates a fixed matrix of GAN configs (varying the DWT discriminator, stacking meta-head, PostNet, and `lambda_fm`) for one subject, to quantify each architectural addition's actual effect on synthetic EEG quality — rather than relying on qualitative/single-checkpoint comparisons.

## Requirements

### Requirement: Config matrix training
The system SHALL train one GAN checkpoint per entry in a fixed ablation config matrix (baseline, +DWT, +DWT+PostNet, PostNet-only, and a `lambda_fm` sweep), for a single chosen subject, using the existing `train_gan()` function unchanged, with identical `patch_size`, `seed`, and `n_epochs` across all entries so only the ablated knob(s) differ.

#### Scenario: Running the ablation matrix
- **WHEN** the ablation script is run for a subject whose CSV already exists
- **THEN** it trains one checkpoint per config entry, each saved under a name that encodes the config (e.g. `ABLATION_<name>_s<subject>.pt`) and does not overwrite checkpoints produced by `moabb_pipeline.py`

### Requirement: Shared metric computation
The system SHALL compute spectral JSD, MMD², and ERP peak amplitude/latency error for each trained config's synthetic samples against the same real held-out trials, using metric functions shared with (not duplicated from) `compare_samples.py`.

#### Scenario: Metrics reused, not reimplemented
- **WHEN** the ablation script computes JSD or MMD² for a config
- **THEN** it calls the same function `compare_samples.py` uses for its own reporting (imported from a shared module), not a separately maintained copy

#### Scenario: ERP peak metric added
- **WHEN** the ablation script evaluates a config
- **THEN** it reports the trial-averaged P300 peak amplitude and latency difference (real vs. synthetic) in addition to JSD and MMD²

### Requirement: Per-config plots in a dedicated subfolder
The system SHALL produce, for each config, the same set of plots `compare_samples.py` already produces for a single checkpoint (ERP overlay per condition, PSD overlay per condition, PCA, t-SNE), written to that config's own subfolder under a dedicated output directory separate from `comparison_plots/`.

#### Scenario: Per-config plots produced
- **WHEN** a config has been trained and evaluated
- **THEN** its ERP, PSD, PCA, and t-SNE plots exist under `ablation_results/<config_name>/`, not mixed with any other config's plots

### Requirement: Combined comparison output
The system SHALL also produce one CSV summarizing all configs' metrics and one figure overlaying all configs' PSD curves per channel, both written at the top level of the same output directory (redundant with, not a replacement for, the per-config plots).

#### Scenario: Summary table produced
- **WHEN** all configs in the matrix have been trained and evaluated
- **THEN** a CSV file exists with one row per config and columns for JSD, MMD², and ERP peak amplitude/latency error

#### Scenario: Combined PSD figure produced
- **WHEN** all configs in the matrix have been evaluated
- **THEN** a single figure exists showing all configs' mean PSD curves overlaid per channel, distinct from each config's own per-condition PSD figure
