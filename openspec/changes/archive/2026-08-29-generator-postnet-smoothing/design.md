## Context

`TTSGenerator` (`eeggan/nn_architecture/tts_gan_components.py::Generator`, subclassed unmodified in `eeggan/nn_architecture/models.py`) reconstructs the output sequence from non-overlapping transformer patch tokens via a single `Conv2d(embed_dim, channels, 1, 1, 0)` projection (`deconv`) followed by a reshape. There is no mechanism smoothing the boundary between adjacent patches. On `main`, this was accidentally masked: the AE-coupled path (`DecoderGenerator`) ran generator output through a frozen `TransformerDoubleAutoencoder` decoder trained only on real-signal reconstruction, which implicitly regularized the output onto a smoother manifold. `ttsgan-native-multichannel` removed that coupling on `ttsgan-direct` because it required decoding fake latents through the AE on every discriminator step just to hand the DWT discriminator (`MultiscaleDWTDiscriminator`, operates on raw samples) something to decompose — an awkward, avoidable dependency between the two discriminators' input requirements.

Confirmed empirically: real BNCI2014_009 P300 trials are bandpass-filtered to `[1, 24]` Hz (`moabb.paradigms.P300().filters == [[1, 24]]`), and PSD comparison plots (`compare_samples.py::plot_psd_overlay`, added this session) show synthetic samples carrying energy above that band that persisted even after raising the DWT feature-matching weight (`lambda_fm`) from 20 to 50 — feature matching alone wasn't a strong enough lever on a problem that's structural to how the generator reconstructs its output.

## Goals / Non-Goals

**Goals:**
- Suppress patch-boundary seam artifacts at the source (inside the generator's forward pass), trained end-to-end with the existing GAN objective — not a post-processing filter applied after generation.
- No coupling to a second network (no AE, no frozen decoder, no extra forward pass needed to feed any discriminator).
- Fully backward compatible: default off, existing checkpoints load and run unchanged.
- Parametrizable through the same CLI-flag pattern already established for `dwt_j` / `lambda_fm` in this codebase, so it's a one-token opt-in (`use_postnet`) at training time, and self-describing at inference time (recorded in the checkpoint's `configuration` dict).

**Non-Goals:**
- Not attempting a rigorous ablation/hyperparameter search over PostNet depth or kernel size in this change — ship a reasonable Tacotron2-inspired default, tune later if the PSD plots ask for it.
- Not reintroducing the autoencoder-coupled generator/discriminator path in any form.
- Not adding an explicit spectral/band-energy loss term (considered as the alternative in conversation; deferred as a separate change if PostNet alone doesn't fully close the gap).
- Not changing `TTSDiscriminator` or `MultiscaleDWTDiscriminator` — they keep seeing whatever the generator (now optionally PostNet-smoothed) hands them; no discriminator-side change needed.

## Decisions

**1. PostNet as a residual `nn.Conv1d` stack, applied over the time axis, inside `TTSGenerator.forward`.**
Modeled on Tacotron2's PostNet, which solves the analogous problem (block/frame-boundary seams in generated mel-spectrograms) the same way: a small residual conv stack refining the raw output, trained jointly with the main network via the existing loss, no separate objective. Alternative considered — overlapping patches with overlap-add reconstruction in the `deconv` stage itself — would fix seams more directly but requires reworking the patchify/de-patchify math inherited from `tts_gan_components.py` (a vendored third-party component per `CLAUDE.md`); PostNet is additive and touches none of that vendored code.

**2. New `PostNet` class lives in `eeggan/nn_architecture/models.py`, not `tts_gan_components.py`.**
Keeps the vendored `imics-lab/tts-gan` component file untouched (per repo convention noted in `CLAUDE.md` §2/§7) and the addition clearly attributable as this project's own. `TTSGenerator` (already a thin subclass in `models.py`) gets `self.postnet` and an overridden `forward` that calls `super().forward(z)` then conditionally applies it.

**3. No `BatchNorm1d` in the PostNet stack — use plain `Conv1d` + `Tanh`, no normalization layer.**
Two independent reasons converge here: (a) nothing else in this codebase's generator/discriminator uses `BatchNorm` — `tts_gan_components.py` uses `LayerNorm` exclusively, and introducing a different norm family for one small module is an unforced inconsistency; (b) `generate_samples_main.py` can run inference with `batch_size=1` (`num_samples_parallel` is user-controlled down to 1), and `BatchNorm1d` in training mode errors on a batch of size 1 — `LayerNorm`/no-norm don't have that failure mode. A small stack (3 conv layers, kernel size 5, residual add) doesn't need normalization to train stably at this scale.

**4. `use_postnet` defaults to `False` everywhere and is recorded in the trainer's `configuration` dict.**
Matches how `lambda_gp`/`lambda_fm`/`dwt_j` are already threaded (`system_inputs.py` schema → `gan_training_main.py opt` dict → consumer, `.get(key, default)` throughout). Recording it in `configuration` (`trainer.py`, next to `lambda_fm`) is required — unlike a loss-weight hyperparameter, `use_postnet` changes the generator's parameter shape/architecture, so `generate_samples_main.py` must know whether to instantiate `TTSGenerator` with the PostNet layer before loading `state_dict`, or the load will fail on a shape/key mismatch.

## Risks / Trade-offs

- **[Risk]** A checkpoint trained with `use_postnet=True` has extra `state_dict` keys (`postnet.*`); loading it while `initialize_gan`/`generate_samples_main` build a `use_postnet=False` generator raises a `state_dict` mismatch. → **Mitigation**: `generate_samples_main.py` must read `use_postnet` from the saved checkpoint `configuration`, not from CLI/default, when reconstructing the generator for inference (already listed in proposal Impact).
- **[Risk]** PostNet might smooth away genuine sharp P300 morphology (the P300 peak itself is a fast transition), not just spurious seams. → **Mitigation**: it's a *residual* addition (`output + postnet(output)`), not a replacement — the base generator output is preserved and the correction is additive; validate post-implementation via the existing `compare_samples.py` ERP overlay (peak shape/latency) alongside the new PSD plot, not PSD alone.
- **[Risk]** Fixed 3-layer/kernel-5 default may be too weak (seams persist) or too strong (over-smooths) for this specific patch_size=10 seam frequency. → **Mitigation**: non-goal to tune now; if `compare_samples.py` PSD plots after training show it's insufficient, revisit depth/kernel as a follow-up rather than blocking this change on a hyperparameter sweep.
- **[Trade-off]** Doesn't address the seam mechanism itself (unlike the overlap-add alternative) — it's corrective, not preventive. Accepted per Decision 1: preventive fix would touch vendored `tts_gan_components.py` patch reconstruction math, larger surface than justified for a first attempt.

## Migration Plan

Purely additive, no migration of existing artifacts needed: default `False` reproduces current behavior byte-for-byte (no `postnet` module instantiated, `forward` unchanged path). Existing checkpoints have no `use_postnet` key in their saved `configuration`; both `gan_training_main.py`/`trainer.py` (writing) and `generate_samples_main.py`/`get_gan_config.py` (reading) must default missing `use_postnet` to `False` via `.get('use_postnet', False)`, so old checkpoints keep loading and generating exactly as before. No rollback procedure needed beyond training with `use_postnet=False` (the default) to skip the new code path entirely.

## Open Questions

- Should the PostNet's depth/kernel size become CLI-parametrizable too, or stay fixed constants inside `PostNet.__init__`? Leaning fixed for now (Non-Goal: no hyperparameter search this change) — only the on/off flag (`use_postnet`) is exposed. Revisit if initial results need tuning.
