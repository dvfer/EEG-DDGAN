## Why

On the `main` branch, training a conditional GAN (`kw_conditions` set) coupled to a `target=full` (two-level `TransformerDoubleAutoencoder`) checkpoint crashes with a shape mismatch inside the AE's channel sub-encoder. `target=full` is currently undocumented-broken and hard-blocked in `train_eeggan_vanilla.sh`, blocking use of the double autoencoder for conditional P300 generation.

## What Changes

- Fix `EncoderDiscriminator`/`TransformerDoubleAutoencoder` coupling so condition columns appended to real/fake data never reach the double AE's fixed-width channel Linear layers (`linear_enc_in_channels` at encode time, `linear_dec_out_channels` at decode time) — split conditions off before `model_2`'s channel step, reattach after, mirroring the existing `target=channels` patch documented in `train_eeggan_vanilla.sh`.
- Remove `train_eeggan_vanilla.sh`'s `target=full` early-exit guard and its "no funciona" comment block; wire `target=full` through the same per-condition training path already used for `time`/`channels`.
- Update `train_eeggan_vanilla.sh`'s usage comment to reflect that all three targets (`time`, `channels`, `full`) are supported.

## Capabilities

### New Capabilities
- `autoencoder-gan-coupling`: correctness contract for coupling a frozen (single- or double-level) autoencoder to a conditional GAN via `DecoderGenerator`/`EncoderDiscriminator` — condition columns must never reach an AE sub-network whose input width was fixed at AE-training time.

### Modified Capabilities
(none — no existing spec currently covers this behavior)

## Impact

- `EEG-DDGAN` `main` branch only (`ttsgan-direct` has no AE-coupled path).
- Code: `eeggan/nn_architecture/models.py` (`EncoderDiscriminator`), `eeggan/nn_architecture/ae_networks.py` (`TransformerDoubleAutoencoder`), `eeggan/helpers/initialize_gan.py` (`init_gan`'s `target=='full'` branch, dimension bookkeeping).
- `EEG-DDGAN/train_eeggan_vanilla.sh` (remove the `target=full` guard, extend `DISC_PATCH`-equivalent handling to the double-AE case).
- No breaking change to `time`/`channels` targets or to `ttsgan-direct`.
