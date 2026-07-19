## Context

EEG-DDGAN's GAN training entrypoint (`gan_training_main.py` → `helpers/initialize_gan.py::init_gan`) currently has two branches: a "no autoencoder" branch that builds a plain `TTSGenerator`/`TTSDiscriminator` pair, and an "autoencoder" branch that wraps them in `DecoderGenerator`/`EncoderDiscriminator` around a `TransformerAutoencoder`/`TransformerDoubleAutoencoder` loaded from a separate pretrained checkpoint, overriding `n_channels`/`sequence_length_generated` from the autoencoder's own output dimensions. EEG-DDGAN's actual research contribution — `MultiscaleDWTDiscriminator` as a secondary discriminator and `StackingDiscriminator` as a meta-learner over both discriminators' features — sits on top of whichever branch runs, and today has only been exercised with the no-autoencoder branch on single-channel EEG (`n_channels=1`).

The user wants to (a) stop carrying the autoencoder branch's complexity going forward, since it was inherited from EEG-GAN and isn't used by the dual-discriminator work, and (b) validate the remaining pipeline against multivariate EEG, which is the eventual target data format. This is scoped as a new git branch inside `EEG-DDGAN` (not a new repo, not a change to `EEG-GAN` upstream).

## Goals / Non-Goals

**Goals:**
- Produce a `ttsgan-direct-pipeline` where `init_gan` only ever constructs plain `TTSGenerator`/`TTSDiscriminator` (+ optional `MultiscaleDWTDiscriminator`/`StackingDiscriminator`) — no `DecoderGenerator`/`EncoderDiscriminator`/`Autoencoder` involvement.
- Preserve WGAN-GP training mechanics (critic iterations, gradient penalty, checkpoint/history format for the parts that don't depend on the autoencoder, DDP entrypoint) exactly as EEG-DDGAN implements them today.
- Preserve the dual-discriminator/stacking behavior exactly as implemented today, just re-parented onto the simplified `init_gan` — this explicitly includes the L1 feature-matching loss term (`lambda_fm=20`) between real and fake discriminator features, in both the plain secondary-discriminator branch and the stacking branch of `GANTrainer.batch_train`. Feature matching is not an implementation detail being dropped along with the autoencoder — it stays.
- Make `n_channels` a validated dimension from `Dataloader` through generator, primary discriminator, and `MultiscaleDWTDiscriminator`, and prove it with a real multi-electrode dataset.
- Isolate all of this on a new branch off EEG-DDGAN so `main` (and EEG-GAN upstream) is untouched.

**Non-Goals:**
- Redesigning the WGAN-GP loss, the transformer architecture, or the wavelet discriminator's internal structure (scale count `J`, MLP head sizing) — this change re-plumbs channel dimensions through them, it does not re-derive them.
- Keeping the autoencoder-conditioned GAN path alive *on this new branch*. It remains available on `main` for anyone who needs it.
- Re-adding VAE/autoencoder training as part of this change's scope — `vae_training_main.py`/`autoencoder_training_main.py` are simply not touched, positively or negatively, beyond no longer being reachable from the new GAN training path's dependency graph.
- Making `MultiscaleDWTDiscriminator` parameter-efficient at large channel counts — flagged as a risk, not solved here.

## Decisions

**1. Isolate via git branch, not a runtime flag.**
Create a new branch off EEG-DDGAN's current default branch (e.g. `ttsgan-direct`) and make the autoencoder branch removal a hard deletion on that branch, rather than adding an `--architecture=direct|autoencoder` flag to `init_gan`/`system_inputs.py`.
*Alternative considered*: keep both paths behind a flag in a single branch. Rejected — the user explicitly asked for a branch, and a flag would mean permanently carrying the autoencoder-adjustment logic (dimension overrides, extra checkpoint loading) as dead weight in every future change to `init_gan`/`trainer.py`, for a feature this line of work does not use. Branches are cheap here; the maintenance cost of a permanent flag is not.

**2. Delete, don't stub, the autoencoder branch.**
On the new branch: remove the `autoencoder != ''` branch from `init_gan`, remove the `--autoencoder` CLI argument from `default_inputs_training_gan()`, and remove the corresponding import/loading code from `gan_training_main.py`. `channel_in_disc`/`sequence_length_generated` are computed solely from the dataloader-derived `n_channels`/`sequence_length`, matching what already happens in the no-autoencoder branch today.
*Alternative considered*: leave the CLI arg accepted but raise `NotImplementedError` if set. Rejected as unnecessary indirection — deleting it is one line shorter and the error a missing keyword produces (`system_inputs.py`'s "Keyword not recognized") already communicates unavailability.

**3. Discriminator2/Stacking wiring is untouched, only its channel inputs change.**
The `kwargs.get('use_multiscale_dwt_discriminator', ...)` / `use_stacking` block at the end of `init_gan` stays as-is; only the `channel_in_disc` value feeding `MultiscaleDWTDiscriminator(in_channels=channel_in_disc, ...)` changes source (dataloader-derived instead of possibly-autoencoder-adjusted).

**3a. Feature-matching loss is carried over verbatim, not reworked.**
The L1 feature-matching blocks in `GANTrainer.batch_train` — the `features_fake_d1`/`features_fake_d2` unpacking, the `real_d1_input`/`out_real` computation under `torch.no_grad()`, and the `lambda_fm * torch.nn.functional.l1_loss(...)` terms added to `g_loss`/`g_loss2` — are copied over unchanged from EEG-DDGAN's current `trainer.py`. The only thing this change may need to touch is confirming the feature tensors' shapes stay consistent when `n_channels > 1` (see Decision 4); the loss formula and weighting are out of scope for modification.
*Alternative considered*: simplify or drop feature matching while removing the autoencoder path, since it adds branching complexity to `batch_train`. Rejected explicitly — the user confirmed feature matching must be kept; it is treated as a fixed part of the dual-discriminator design this change re-parents, not something up for renegotiation here.

**4. Validate multichannel with explicit shape checks + one real integration run, not a new test suite.**
Add targeted `assert`-based shape checks at the three join points that are new risk surface for `n_channels > 1`: (a) `Dataloader` output going into the generator's conditioning/labels concatenation, (b) generator output → both discriminators' expected `in_channels`, (c) `WassersteinGradientPenaltyLoss._gradient_penalty`'s `(B, C, 1, L)` reshape. Follow with one short real training run (few epochs) on an actual multi-electrode dataset as the acceptance check, rather than building a synthetic-data pytest suite — this is exploratory research code, and the existing `eeggan/tests/test_*.py` scripts are already this style (small `if __name__ == '__main__'` smoke configs, not pytest).
*Alternative considered*: full pytest coverage of multichannel shapes. Rejected as disproportionate to a single-developer research codebase at this stage; can be added later if the pipeline stabilizes.

## Risks / Trade-offs

- **[Risk]** Deleting the autoencoder branch on this branch makes it impossible to load/resume old EEG-DDGAN checkpoints trained with an autoencoder-wrapped generator/discriminator, on this branch. → **Mitigation**: this capability stays fully available on `main`; `generate_samples_main.py`/inference for existing autoencoder-based checkpoints is unaffected there. This branch is additive exploration, not a replacement until proven out.
- **[Risk]** `MultiscaleDWTDiscriminator`'s per-scale MLP heads have input size `approx_len * in_channels` (and `detail_len * in_channels` for high-freq) — parameter count grows linearly with channel count per scale head, which could get large for high electrode counts. → **Mitigation**: validate first at a moderate channel count (single digits to low tens of electrodes); treat `hidden_dim`/`J`/`multiscale_dwt_high_freq` as tunable if parameter count or training stability becomes a problem, rather than redesigning the discriminator up front.
- **[Risk]** `WassersteinGradientPenaltyLoss._gradient_penalty` reshapes real/fake tensors to `(B, C, 1, L)` via `permute` — this logic has only been exercised at `C=1`; a silent broadcasting mistake at `C>1` would corrupt the gradient penalty without raising an error. → **Mitigation**: explicit shape assertions before/after this reshape during implementation (per Decision 4), not just trusting it "probably still works."
- **[Risk]** `GANDDPTrainer.set_ddp_framework` wraps `generator`/`discriminator` in `DDP(...)` but not `discriminator2` — multi-GPU behavior for the dual-discriminator setup is unverified. → **Mitigation**: flagged as an open question below rather than silently assumed to work; DDP correctness for `discriminator2` is out of scope unless explicitly picked up.

## Migration Plan

1. Branch off EEG-DDGAN's current default branch.
2. Remove the autoencoder branch from `init_gan`/`gan_training_main.py`/`system_inputs.py` (Decisions 1–3).
3. Regression-check: re-run existing single-channel training config (no autoencoder, with `use_multiscale_dwt_discriminator`/`use_stacking`) and confirm loss curves/checkpoint shape match pre-change behavior — this is the safety net proving the deletion didn't change behavior for the case that already worked.
4. Add the multichannel shape assertions (Decision 4a–c).
5. Run the short real multi-electrode training integration check (Decision 4).
6. No rollback mechanism needed beyond not merging/deleting the branch — `main` is never touched.

## Open Questions (resolved during implementation)

- **Should `vae_training_main.py`/`autoencoder_training_main.py` be deleted from this branch outright, or just left present-but-unused?** → Resolved: left present-but-unused. They're independent CLI commands (`eeggan vae_training`, `eeggan autoencoder_training`) that don't import from or get imported by anything touched in section 2; deleting them would be an unrelated destructive change outside this proposal's stated Non-Goals ("not re-adding/removing VAE/autoencoder training as part of this change's scope").
- **What channel/electrode count should the first multichannel validation run target, and which real dataset will be used?** → Resolved by discovery, not decision: `origin/mv-train`'s `moabb_pipeline.py` already targets `BNCI2014_009` (P300), whose channel count is whatever that dataset's EEG montage provides (determined at data-export time, not hardcoded). This is the dataset task 5 will validate against; the exact channel count is confirmed when the pipeline actually runs (see `notes.md`).
- **Is DDP support for `discriminator2`/`StackingDiscriminator` in scope for this change, or explicitly deferred to a follow-up?** → Resolved: deferred. `GANDDPTrainer.set_ddp_framework` still only wraps `generator`/`discriminator` in `DDP(...)`, not `discriminator2`; this change does not touch DDP code. Multi-GPU training with the dual discriminator remains unverified and is left as a follow-up.
