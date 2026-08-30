## 1. Set up main-branch worktree

- [x] 1.1 Create/reuse `../EEG-DDGAN-main-vanilla` git worktree of `main` (same one `train_eeggan_vanilla.sh` uses)

## 2. Fix EncoderDiscriminator

- [x] 2.1 In `eeggan/nn_architecture/models.py`, add `n_conditions: int = 0` to `EncoderDiscriminator.__init__`, stored as `self.n_conditions`
- [x] 2.2 In `EncoderDiscriminator.forward`, when `self.n_conditions > 0`: split `input_data` into `raw = input_data[..., :-self.n_conditions]` / `cond = input_data[..., -self.n_conditions:]`, call `self.encoder.encode(raw)`, concatenate `cond` back onto the encoded result along the last dim before calling `self.discriminator(...)`; when `self.n_conditions == 0`, keep current behavior unchanged (refined during verification: `cond` must be re-broadcast to `encoded`'s sequence length, not sliced at the original length — see design.md)
- [x] 2.3 In `eeggan/helpers/initialize_gan.py`, pass `n_conditions=n_conditions` when constructing `EncoderDiscriminator` in the autoencoder branch

## 3. Regression-check target=channels

- [x] 3.1 Re-run the dummy-tensor repro (generator+decode, discriminator+encode) for `target=channels`, conditional and unconditional, confirm shapes match pre-fix behavior
- [x] 3.2 Remove the now-redundant inline `DISC_PATCH` monkeypatch block from `train_eeggan_vanilla.sh` (the fix lives in `models.py` now)

## 4. Verify target=full via repro

- [x] 4.1 Re-run the dummy-tensor repro for `target=full`, conditional (`n_conditions=1`) and unconditional, confirm no `RuntimeError` and correct output shapes end-to-end (generator→decode, discriminator→encode)

## 5. Unblock target=full in the training script

- [x] 5.1 Remove the `target=full` early-exit guard and its "no funciona" comment block from `train_eeggan_vanilla.sh`
- [x] 5.2 Update the script's usage/header comments to state all three targets (`time`, `channels`, `full`) are supported
- [x] 5.3 Update `AE_NAME`/`GAN_NAME` naming and any target-specific branches (if the `channels`/`full` disc patch is now unconditional and no longer needs the `DISC_PATCH` heredoc injection, simplify the GAN-training heredoc accordingly)

## 6. End-to-end validation on GPU box

- [ ] 6.1 Run `./train_eeggan_vanilla.sh 1 full` on the GPU box (demerzel) against real subject data end-to-end (AE training → GAN training → sample generation)
- [ ] 6.2 Spot-check generated samples (shape, no NaNs) and confirm `eval_external_config.py` can ingest the output CSV
