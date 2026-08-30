## 1. Model: PostNet module + TTSGenerator wiring

- [x] 1.1 In `eeggan/nn_architecture/models.py`, add a `PostNet(nn.Module)` class: a small residual `Conv1d` stack over the time axis (few layers, kernel size ~5, `Tanh` activations, no `BatchNorm1d`/no norm layer — see design.md Decision 3), input/output shape `(B, L, C)`.
- [x] 1.2 In `TTSGenerator.__init__` (`models.py`), add a `use_postnet=False` parameter; instantiate `self.postnet = PostNet(channels)` only when `True` (leave `self.postnet = None` otherwise, matching the codebase's existing `discriminator2 = None` pattern).
- [x] 1.3 Override `forward` on `TTSGenerator` to call the parent's `forward(z)` then, if `self.postnet is not None`, add its output as a residual before returning. When `use_postnet=False`, output must be identical to pre-change behavior (spec: "PostNet disabled (default)").

## 2. Wiring: generator construction and CLI

- [x] 2.1 In `eeggan/helpers/initialize_gan.py`, the `gan_architectures['TTSGenerator']` lambda currently accepts `**kwargs` but drops them before constructing `TTSGenerator(...)` — extend it to accept and forward `use_postnet`.
- [x] 2.2 In `init_gan()` (`initialize_gan.py`), read `use_postnet = kwargs.get('use_postnet', False)` and pass it through to the `gan_architectures[...]( ... )` generator-construction call (that call currently only forwards specific named args, not `init_gan`'s own `**kwargs`).
- [x] 2.3 In `eeggan/helpers/system_inputs.py`, add a `use_postnet` bool flag to `default_inputs_training_gan()`, same list-entry pattern as `use_multiscale_dwt_discriminator` / `dwt_j`, default `False`.

## 3. Training: opt dict + checkpoint configuration

- [x] 3.1 In `eeggan/gan_training_main.py`, add `'use_postnet': default_args.get('use_postnet', False)` to the `opt = {...}` dict (same place `dwt_j`/`lambda_fm` were added) so it reaches `init_gan(**opt)`.
- [x] 3.2 In `eeggan/helpers/trainer.py`, add `'use_postnet': opt.get('use_postnet', False)` to `GANTrainer`'s `self.configuration` dict (next to `lambda_fm`) so every saved checkpoint self-describes this.

## 4. Inference: generate_samples_main.py

- [x] 4.1 In `eeggan/generate_samples_main.py`, add `use_postnet=state_dict['configuration'].get('use_postnet', False),` to the `init_gan(...)` call (~line 122-131), so the reconstructed generator matches the checkpoint's actual trained architecture regardless of CLI args passed to `generate_samples`.
- [x] 4.2 Verify `eeggan/get_gan_config.py` needs no code change — it prints `state_dict['configuration']` generically, so `use_postnet` will show up automatically once 3.2 is done. Just confirm by running it against a checkpoint saved before this change (missing key must not crash `generate_samples`/`get_gan_config`).

## 5. moabb_pipeline.py exposure

- [x] 5.1 Add `GAN_USE_POSTNET = False` to the GAN config block (next to `GAN_USE_DWT`/`GAN_DWT_J`/`GAN_LAMBDA_FM`).
- [x] 5.2 Add `use_postnet=False` parameter to `train_gan()` and append `'use_postnet'` (bare flag) to its CLI `args` list when `True`.
- [x] 5.3 Pass `use_postnet=GAN_USE_POSTNET` at the `train_gan(...)` call site in `main()`.

## 6. Verification

- [x] 6.1 Add a small `__main__`/`demo()` self-check (in `models.py` or a standalone script) asserting: (a) `TTSGenerator(..., use_postnet=False)` output shape and values match a build without the `use_postnet` arg at all (backward-compat no-op); (b) `TTSGenerator(..., use_postnet=True)` output shape is unchanged `(B, seq_len, channels)` and gradients reach `postnet.*` parameters after a dummy `.backward()`. Ran via `uv run python -m eeggan.nn_architecture.models` — passes.
- [x] 6.2 End-to-end smoke test: run `eeggan gan_training ... use_postnet n_epochs=1` on a small CSV and confirm training completes and the saved checkpoint's `configuration['use_postnet']` is `True`. Ran against a synthetic throwaway CSV — checkpoint saved with `use_postnet: True` and `postnet.*` weights present; smoke-test checkpoint deleted after.
- [x] 6.3 Backward-compat smoke test: run `eeggan generate_samples` against an existing pre-change checkpoint (no `use_postnet` key) and confirm it still loads and generates without error. Verified by stripping `use_postnet`/`postnet.*` from the 6.2 checkpoint to simulate a pre-change save — `generate_samples` loaded and generated fine (defaults to `False`).
- [x] 6.4 Re-run `compare_samples.py` (with `plot_psd_overlay`) on a `use_postnet=True` checkpoint vs. the existing `use_postnet=False` one for the same subject, and eyeball whether high-frequency energy above 24 Hz visibly drops. Done via `openspec/changes/ablation-study-harness/` (`ablation_pipeline.py`, subject 1, real training): `baseline` (no postnet, no DWT) JSD=0.286 vs `postnet_only` (postnet, no DWT) JSD=0.0064 — PostNet alone cuts spectral JSD by ~45×, confirming the design works. Caveat found: PostNet's benefit depends heavily on `lambda_fm` — `postnet+lambda_fm=0` is one of the worst configs in the full matrix (JSD 0.0068), while `postnet+lambda_fm=50` is the best (JSD 0.0011) — PostNet needs a strong feature-matching signal to train well, it isn't a free win on its own once DWT is also in play. Full numbers/discussion in `ablation-study-harness`'s results.
