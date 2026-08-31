## Context

`init_gan()` (`main` branch) can couple a frozen autoencoder to the TTS-GAN generator/discriminator via `DecoderGenerator`/`EncoderDiscriminator`. For `target=channels`, this already required a patch (documented in `train_eeggan_vanilla.sh`'s `DISC_PATCH`): `EncoderDiscriminator.forward` calls `self.encoder.encode(data)` on the raw discriminator input, which is `[real_signal | condition_columns]` concatenated on the channel axis. A `channels`-target AE's encoder has a `Linear(channels_in, ...)` first layer sized at AE-training time (no condition columns present then) — so it breaks unless conditions are split off first and reattached after encoding.

`target=full` (`TransformerDoubleAutoencoder`) has the identical exposure, one level deeper: `model_1` (timeseries sub-AE) is channel-count-agnostic and passes conditioned input through fine, but `model_2`'s channel sub-encoder/decoder (`linear_enc_in_channels`, `linear_dec_out_channels`) are `Linear(channels_in, ...)` / `Linear(..., channels_in)`, sized the same way. Confirmed by direct repro against `main`'s real modules: unconditional `target=full` runs clean end-to-end; conditional `target=full` crashes inside `TransformerDoubleAutoencoder.encode()` at `linear_enc_in_channels` with a width mismatch (`channels_in` vs `channels_in + n_conditions`). This is currently worked around by hard-blocking `target=full` in `train_eeggan_vanilla.sh` rather than fixed.

## Goals / Non-Goals

**Goals:**
- Let a conditional GAN train coupled to a `target=full` autoencoder without the AE's channel sub-network ever seeing condition columns.
- Reuse the same split-before/concat-after shape as the existing `target=channels` `DISC_PATCH`, rather than inventing a second mechanism.
- Unblock `target=full` in `train_eeggan_vanilla.sh` once the underlying fix lands.

**Non-Goals:**
- Touching `ttsgan-direct` (no AE-coupled path exists there; out of scope).
- Changing AE training itself (`autoencoder_training_main.py`'s two-level training loop is already correct and unconditional — confirmed via repro).
- Adding native `kw_conditions` support to `autoencoder_training_main.py` (conditions are only ever added downstream, at GAN-coupling time, same as today).

## Decisions

**Patch `EncoderDiscriminator.forward`, not `TransformerDoubleAutoencoder.encode`/`decode`.**
The condition columns are a GAN-training-time concern (`EncoderDiscriminator` glues them on); the AE itself has no notion of conditions and shouldn't need one. Splitting/reattaching in `EncoderDiscriminator.forward` — generalizing it to strip the last `n_conditions` channel columns before calling `self.encoder.encode(...)` and reattach them to the encoded result before calling `self.discriminator(...)` — fixes both `target=channels` and `target=full` at a single call site, and matches what the existing inline `DISC_PATCH` in `train_eeggan_vanilla.sh` already does ad hoc for `target=channels`. This makes the patch a permanent part of `main`'s `models.py` instead of a training-script-only workaround.
Alternative considered: teach `TransformerDoubleAutoencoder`/`TransformerAutoencoder` to accept and ignore a conditions width. Rejected — it couples the AE's architecture to GAN-training concerns it doesn't otherwise have, and would require passing `n_conditions` through AE construction/checkpointing.

**Only the channel-facing sub-network needs the split; time-facing paths pass through unchanged.**
`model_1`/time-target encode-decode already tolerate any channel count (transformer treats channels as a free sequence-length axis). So the split only needs to happen once, around the whole `encoder.encode()`/`encoder.decode()` call in `EncoderDiscriminator`/`DecoderGenerator` — no need to reach inside `TransformerDoubleAutoencoder` to special-case `model_2`.

**Reattached conditions must be re-broadcast to `encoded`'s sequence length, not sliced at the original length.** Discovered via the verify-fix repro: for `target=full`, `encode()` changes the sequence axis too (`model_1` goes `time_in → time_out`), so the condition slice (still `time_in`-long) can't be concatenated onto the shorter encoded tensor by shape alone. Since a condition value is constant across the whole sequence axis (broadcast at dataset-export time), the fix takes `cond[:, :1, :]` and re-expands it to `encoded.shape[1]` rather than reusing the original slice — a no-op for `channels`/`time` targets (sequence length unchanged there) and correct for `full`. This is the real mechanism behind the "50 vs 64 mismatch" the old script comment blamed on target=full being fundamentally broken.

**`DecoderGenerator` needs no change.**
The generator's decoded output is the raw reconstructed signal with no condition columns (conditions only enter as generator *input* latent concatenation, handled elsewhere in `gan_training_main.py`); confirmed by repro (`generator+decode output` shape matches raw `channels_in`, no condition column). Only the discriminator side glues conditions onto its *input*.

## Risks / Trade-offs

- [Generalizing `EncoderDiscriminator.forward` changes behavior for `target=channels` too, not just `target=full`] → Low risk: it becomes the same split/reattach `DISC_PATCH` already exercises today, just moved from a training-script monkeypatch into the real class; verify with a repro-based regression check for `target=channels` before/after (same technique used to find this bug) as a tasks.md step.
- [`n_conditions` isn't currently plumbed into `EncoderDiscriminator.__init__`] → Pass it through from `init_gan()`, where `n_conditions` is already an argument; default `0` preserves current unconditional behavior for any other caller.
- [Double AE case (`target=full`) has never been exercised end-to-end with real training data] → tasks.md includes an actual GPU-box run of `train_eeggan_vanilla.sh 1 full` as the final verification step, not just the dummy-tensor repro.

## Migration Plan

No data/checkpoint migration — existing `time`/`channels` checkpoints and unconditional `full` checkpoints are unaffected (the split becomes a no-op when `n_conditions=0`). Rollback is a single-commit revert on `main` if the GPU-box run surfaces a further issue.
