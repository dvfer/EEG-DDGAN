import torch
from torch import nn

from eeggan.nn_architecture.ae_networks import Autoencoder
from eeggan.nn_architecture.tts_gan_components import Generator as TTSGenerator_Org, Discriminator as TTSDiscriminator_Org

# insert here all different kinds of generators and discriminators
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, z):
        raise NotImplementedError


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, z):
        raise NotImplementedError


class TTSGenerator(TTSGenerator_Org):
    def __init__(self, seq_len=150, patch_size=15, channels=3, num_classes=9, latent_dim=100, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(TTSGenerator, self).__init__(seq_len, patch_size, channels, num_classes, latent_dim, embed_dim, depth,
                 num_heads, forward_drop_rate, attn_drop_rate)


class TTSDiscriminator(TTSDiscriminator_Org):
    def __init__(self, in_channels=3, patch_size=15, emb_size=50, seq_length=150, depth=3, n_classes=1, **kwargs):
        super(TTSDiscriminator, self).__init__(in_channels, patch_size, emb_size, seq_length, depth, n_classes, **kwargs)


class DecoderGenerator(Generator):
    """
    DecoderGenerator serves as a wrapper for a generator.
    It takes the output of the generator and passes it to a given decoder if the corresponding flag was set.
    Otherwise, it returns the output of the generator.
    """

    def __init__(self, generator: Generator, decoder: Autoencoder):
        """
        :param generator: generator model
        :param decoder: autoencoder model that has a decode method
        """
        super(DecoderGenerator, self).__init__()
        self.generator = generator
        self.decoder = decoder
        self.decode = True

        # add attributes from generator
        self.latent_dim = generator.latent_dim if hasattr(generator, 'latent_dim') else None
        self.channels = generator.channels if hasattr(generator, 'channels') else None
        self.seq_len = generator.seq_len if hasattr(generator, 'seq_len') else None

    def forward(self, data):
        if self.decode:
            return self.decoder.decode(self.generator(data))
        else:
            return self.generator(data)

    def decode_output(self, mode=True):
        self.decode = mode


class EncoderDiscriminator(Discriminator):
    """
    EncoderDiscriminator serves as a wrapper for a discriminator.
    It takes the input of the discriminator and passes it to a given encoder if the corresponding flag was set.
    Otherwise, it returns the output of the discriminator.
    """

    def __init__(self, discriminator: Discriminator, encoder: Autoencoder, n_conditions: int = 0):
        """
        :param discriminator: discriminator model
        :param encoder: autoencoder model that has an encode method
        :param n_conditions: number of condition columns concatenated onto the channel axis of
            `data` in forward(). The encoder's channel-facing layers are sized from AE-training
            time (no condition columns present then), so conditions are split off before encode()
            and reattached to the encoded result -- otherwise encode() breaks on the width mismatch.
        """
        super(EncoderDiscriminator, self).__init__()
        self.discriminator = discriminator
        self.encoder = encoder
        self.encode = True
        self.n_conditions = n_conditions

        # add attributes from discriminator
        self.channels = discriminator.channels if hasattr(discriminator, 'channels') else None
        self.n_classes = discriminator.n_classes if hasattr(discriminator, 'n_classes') else None

    def forward(self, data):
        if self.encode:
            input_data = data
            if input_data.dim() == 4:
                input_data = input_data.permute(0, 3, 2, 1).squeeze(2)
            if self.n_conditions > 0:
                raw, cond = input_data[..., :-self.n_conditions], input_data[..., -self.n_conditions:]
                encoded = self.encoder.encode(raw)
                # cond is constant across the sequence axis (same condition value at every timestep);
                # encode() may itself change the sequence length (e.g. target=full's timeseries sub-AE
                # goes time_in -> time_out), so re-broadcast from a single step instead of assuming the
                # original length still matches.
                cond = cond[:, :1, :].expand(-1, encoded.shape[1], -1)
                encoded = torch.cat([encoded, cond], dim=-1)
            else:
                encoded = self.encoder.encode(input_data)
            return self.discriminator(encoded)
        else:
            return self.discriminator(data)

    def encode_input(self, mode=True):
        self.encode = mode


from pytorch_wavelets import DWT1DForward

class MultiscaleDWTDiscriminator(nn.Module):
    """
    Multi-scale DWT discriminator with configurable frequency focus.    
    Uses progressive DWT decomposition and processes coefficients
    at each level with separate MLP heads.
    """
    def __init__(self, in_channels=1, wave='db4', mode='zero', J=4, hidden_dim=64, 
                 n_classes=1, seq_len=150, include_high_freq=True):
        super(MultiscaleDWTDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.J = J
        self.wave = wave
        self.mode = mode
        self.n_classes = n_classes
        self.include_high_freq = include_high_freq
        
        self.dwt = DWT1DForward(J=1, wave=wave, mode=mode)
        
        # Calculate dimensions for each scale level
        self.scale_dims_low = []  # Approximation (low-freq) dimensions
        self.scale_dims_high = []  # Detail (high-freq) dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, seq_len)
            current = dummy_input
            for j in range(J):
                yl, yh = self.dwt(current)
                # yl is low-frequency approximation: (B, C, L_approx)
                # yh is list of high-frequency detail: [(B, C, L_detail)]
                approx_len = yl.shape[-1]
                detail_len = yh[0].shape[-1] if len(yh) > 0 else 0
                self.scale_dims_low.append(approx_len * in_channels)
                self.scale_dims_high.append(detail_len * in_channels)
                current = yl  # Continue with approximation for next level
        
        self.activation = nn.LeakyReLU(0.2)
        
        # Create MLP head for each scale level (low-freq)
        self.scale_heads_low = nn.ModuleList()
        for j in range(J):
            head = nn.Sequential(
                nn.Linear(self.scale_dims_low[j], hidden_dim),
                self.activation,
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                self.activation,
            )
            self.scale_heads_low.append(head)
        
        # Create MLP head for each scale level (high-freq) if enabled
        self.scale_heads_high = nn.ModuleList()
        if include_high_freq:
            for j in range(J):
                head = nn.Sequential(
                    nn.Linear(self.scale_dims_high[j], hidden_dim),
                    self.activation,
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    self.activation,
                )
                self.scale_heads_high.append(head)
        
        # Final combination layer
        num_feature_streams = 2 if include_high_freq else 1
        combined_dim = (hidden_dim // 2) * J * num_feature_streams
        self.final_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x):
        # x shape: (B, L, C) or (B, C, L) or (B, C, 1, L) from GP loss
        
        if x.dim() == 4:
            if x.shape[2] == 1:
                x = x.squeeze(2)
        
        # For DWT1D, we need (N, C, L)
        if x.shape[-1] == self.in_channels:
             x = x.permute(0, 2, 1)
        
        batch_size = x.shape[0]
        
        # Process each scale level progressively
        scale_features = []
        current = x
        
        for j in range(self.J):
            yl, yh = self.dwt(current)
            # yl: approximation (low-frequency) coefficients
            # yh: detail (high-frequency) coefficients
            
            # Process low-frequency approximation
            approx_flat = yl.reshape(batch_size, -1)
            scale_out_low = self.scale_heads_low[j](approx_flat)
            scale_features.append(scale_out_low)
            
            # Process high-frequency detail if enabled
            if self.include_high_freq and len(yh) > 0:
                detail_flat = yh[0].reshape(batch_size, -1)
                scale_out_high = self.scale_heads_high[j](detail_flat)
                scale_features.append(scale_out_high)
            
            # Continue decomposition on approximation
            current = yl
        
        # Concatenate all scale features
        combined_features = torch.cat(scale_features, dim=1)
        
        # Final prediction
        validity = self.final_head(combined_features)
        
        # Return validity and features for feature matching
        return validity, combined_features


class StackingDiscriminator(nn.Module):
    def __init__(self, primary_disc, secondary_disc=None, primary_feat_dim=50, secondary_feat_dim=100, hidden_dim=64):
        super().__init__()
        self.primary = primary_disc
        self.secondary = secondary_disc
        
        # Calculate combined dimension
        self.combined_dim = primary_feat_dim + secondary_feat_dim
        
        # Meta-learner: Combines features
        self.meta_head = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1) # Output: Final Validity
        )
        
    def forward(self, x, x_raw=None):
        # D1 Forward
        out1 = self.primary(x)
        val1, feats1 = out1 if isinstance(out1, tuple) else (out1, None)
        
        val2 = None
        feats2 = None
        
        # D2 Forward
        if self.secondary is not None:
             input2 = x_raw if x_raw is not None else x
             out2 = self.secondary(input2)
             val2, feats2 = out2 if isinstance(out2, tuple) else (out2, None)
             
        # Meta Combination Logic
        if self.secondary is not None and feats2 is not None and feats1 is not None:
             # If Secondary is ACTIVE and features are available:
             # Combine features and use Meta-Head
             # Check if features are flat
             if feats1.dim() > 2:
                  feats1 = feats1.flatten(1)
             if feats2.dim() > 2:
                  feats2 = feats2.flatten(1)
                  
             combined_input = torch.cat([feats1, feats2], dim=1)
             final_validity = self.meta_head(combined_input)
             return final_validity, (feats1, feats2)
        else:
             # If Secondary is DISABLED or features missing:
             # Bypass Meta-Head completely. Return Primary validity.
             return val1, (feats1, None)