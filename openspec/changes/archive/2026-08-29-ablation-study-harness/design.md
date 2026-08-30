## Context

`compare_samples.py` already implements every metric needed (`spectral_jsd`, `mmd_rbf`, Welch PSD, ERP overlay) for a single trained checkpoint vs. real data. `moabb_pipeline.py` already implements `train_gan()`, the one place that knows how to invoke `eeggan gan_training` with the right CLI tokens. The ablation harness doesn't need new training or metric logic — it needs to call both, several times with different config knobs, and tabulate the result.

## Goals / Non-Goals

**Goals:**
- Train the 5-config matrix from the proposal for one subject with a shared epoch/patch/seed budget, varying only the ablated knob(s) per run.
- Reuse `train_gan()` (moabb_pipeline.py) and `generate_synthetic()`/`load()` (compare_samples.py) unchanged — no parallel training/generation code path.
- Produce a single comparison table (CSV) and a single overlay figure (all configs' PSD on one plot per channel) suitable for dropping into the thesis.

**Non-Goals:**
- No comparison against the autoencoder-based `main` branch or the upstream `tts-gan` repo (different checkpoint formats — see proposal's "out of scope").
- No hyperparameter search / automatic epoch-budget tuning — the epoch budget is a fixed constant the user sets, same for every config in the matrix.
- No new metric beyond JSD, MMD², and ERP peak amplitude/latency — no new statistical framework.

## Decisions

**D1 — Extract shared metrics into `eeg_metrics.py` (new top-level module).**
`spectral_jsd`, `mmd_rbf`, and the Welch-PSD computation currently live inline in `compare_samples.py`. Move them (unchanged logic) into `eeg_metrics.py`; `compare_samples.py` imports them back. Both `compare_samples.py` and the new `ablation_pipeline.py` then call the same code — no duplicated metric math to keep in sync.
*Alternative considered:* duplicate the ~30 lines into the ablation script. Rejected — this is exactly the kind of drift (as already happened between the notebook and `compare_samples.py` this session) that causes silent methodology mismatches later.

**D1b — Refactor `compare_subject()` into a reusable, parametrized, metrics-returning function.**
`compare_subject(subject)` currently hardcodes its paths (derived from `MODEL_PREFIX`/`GAN_DIR`/`PLOT_DIR`) and only *prints* JSD/MMD² rather than returning them. Generalize its signature to `compare_subject(subject, model_path=None, real_csv=None, gen_csv=None, plot_dir=None)` (each `None` defaulting to today's derived path, so `compare_samples.py`'s own CLI behavior is unchanged), and make it **return** a metrics dict (`{'jsd_mean', 'mmd2', 'erp_amp_err', 'erp_lat_err_ms'}` per condition or averaged) in addition to its existing prints and plot files. The ablation script then calls this one function per config instead of re-deriving generation/plotting/metric logic — it already produces every per-config plot (ERP overlay ×2 conditions, PSD overlay ×2 conditions, PCA, t-SNE), so per-config outputs come for free by pointing `plot_dir` at that config's own subfolder.
*Alternative considered:* have the ablation script call `generate_synthetic`/`load`/the individual plot functions itself, keeping `compare_subject` untouched. Rejected once per-config plots were requested — that would duplicate exactly the orchestration `compare_subject` already does.

**D2 — Add one new metric: ERP peak amplitude/latency error.**
Add `erp_peak_metrics(real_trials, gen_trials, fs, window_ms=(250, 600))` to `eeg_metrics.py`: finds each condition's trial-averaged peak amplitude and latency inside the P300 window (`argmax` of the grand-average signal within the window, same as the notebook's `erp_peak()`), returns `(amp_error, latency_error_ms)` between real and synthetic. Window is `(250, 600)` ms — taken directly from `signal_analysis.ipynb`'s `P300_WIN` constant (confirmed by reading the notebook; not the initially-assumed 250-500ms).

**D3 — Ablation config matrix as a plain list of dicts, in-script.**
```python
ABLATIONS = [
    {'name': 'baseline',              'use_dwt': False, 'use_stacking': False, 'use_postnet': False},
    {'name': 'postnet_only',          'use_dwt': False, 'use_stacking': False, 'use_postnet': True},
    {'name': 'dwt',                   'use_dwt': True,  'use_stacking': False, 'use_postnet': False, 'lambda_fm': 20},
    {'name': 'dwt_stacking',          'use_dwt': True,  'use_stacking': True,  'use_postnet': False, 'lambda_fm': 20},
    # postnet x lambda_fm factorial (use_dwt=True, use_stacking=False throughout)
    {'name': 'fm_lambda_0',           'use_dwt': True,  'use_stacking': False, 'use_postnet': False, 'lambda_fm': 0},
    {'name': 'fm_lambda_20_postnet',  'use_dwt': True,  'use_stacking': False, 'use_postnet': True,  'lambda_fm': 20},  # == dwt_postnet
    {'name': 'fm_lambda_50',          'use_dwt': True,  'use_stacking': False, 'use_postnet': False, 'lambda_fm': 50},
    {'name': 'fm_lambda_0_postnet',   'use_dwt': True,  'use_stacking': False, 'use_postnet': True,  'lambda_fm': 0},
    {'name': 'fm_lambda_50_postnet',  'use_dwt': True,  'use_stacking': False, 'use_postnet': True,  'lambda_fm': 50},
]
```
9 configs total. `fm_lambda_20_postnet` doubles as the "full model" (DWT + PostNet + default `lambda_fm=20`) — no separate `dwt_postnet` entry needed, it's the middle point of the factorial. `dwt` vs `dwt_stacking` isolates the stacking meta-head's effect (same DWT discriminator, only the combination method differs). The `lambda_fm ∈ {0, 20, 50}` × `use_postnet ∈ {False, True}` factorial (6 of the 9 configs) directly answers whether feature-matching weight matters independently of PostNet, or only in its absence — motivated by the earlier empirical finding that `lambda_fm=50` alone didn't remove high-frequency artifacts.

Same flat-constants style as `moabb_pipeline.py` — no YAML/JSON config files, no CLI framework for defining the matrix. `lambda_fm` only matters when `use_dwt=True` (matches existing `train_gan()` behavior, which only appends it in that branch).

**D4 — One subject, one CSV, shared budget across all configs.**
All 6 runs use the same `ABLATION_SUBJECT`, the same already-exported CSV, the same `patch_size`/`seed`/`n_epochs`. Only the knobs in the matrix vary, so any metric difference is attributable to the architecture change, not to different data or training length.

**D5 — Distinct checkpoint/CSV names per config, separate from the main pipeline's outputs.**
`trained_models/ABLATION_<name>_s<subject>.pt` and `generated_samples/ABLATION_<name>_s<subject>_synthetic.csv` — so this never overwrites the user's already-trained `GAN_009_Modded_*` checkpoints from `moabb_pipeline.py`.

**D6 — Output: per-config subfolder with full plots, plus one combined summary CSV + one combined PSD figure at the top level.**
```
ablation_results/
  <config_name>/                     # one per config, from compare_subject(plot_dir=...)
    erp_Target.png  erp_NonTarget.png
    psd_Target.png  psd_NonTarget.png
    pca.png  tsne.png
  ablation_s<subject>.csv             # combined: one row per config (config, jsd_mean, mmd2, erp_amp_err, erp_lat_err_ms)
  psd_comparison_s<subject>.png       # combined: all 9 configs' PSD overlaid per channel, for the thesis figure
```
Per-config plots are intentionally redundant with the combined CSV/figure — they're for drilling into one config, the combined outputs are for the cross-config comparison. Both come from the same underlying data (no separate computation path): per-config plots from D1b's refactored `compare_subject()`, the combined PSD figure from the same `compute_psd()` calls, just plotted together instead of separately.

## Risks / Trade-offs

- **9 full GPU training runs is real compute cost, not a code-complexity cost** → mitigate by making `ABLATION_N_EPOCHS` a separate, independently-tunable constant from `moabb_pipeline.py`'s `GAN_N_EPOCHS` (start smaller, e.g. a fraction of production budget, raise once results look directionally sane).
- **Reduced epoch budget may leave models under-converged, making metric differences noisy** → mitigated by D4 (identical budget across configs keeps the comparison apples-to-apples even if not fully converged; under-convergence would blur all configs equally, not bias one over another).
- **`lambda_fm=0` may behave differently from "no feature-matching loss at all"** (it still runs the DWT discriminator/feature-matching code path, just multiplied by zero) → acceptable, this is exactly the ablation question being asked (does the weight matter, all else equal), not a claim that D2 is fully absent.

## Open Questions

(none — the P300 peak window was confirmed against `signal_analysis.ipynb` at implementation time: `(250, 600)` ms, see D2.)
