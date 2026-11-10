import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Encode images as a sequence of convolutional pixel features."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64):
        """Initialize the model layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        """Flatten convolutional feature maps into an ordered feature sequence."""
        features = self.net(x)
        batch, channels, height, width = features.shape
        return features.view(batch, channels, height * width).permute(0, 2, 1)


class SlotAttention(nn.Module):
    """Group image features into slots through iterative attention and GRU updates."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ):
        """Initialize the model layers."""
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, init_noise=None, return_attention: bool = False):
        """Refine initialized slots and optionally return their feature attention."""
        batch, _, dim = inputs.shape
        inputs = self.norm_input(inputs)

        mu = self.slots_mu.expand(batch, self.num_slots, -1)
        sigma = torch.exp(self.slots_log_sigma).expand_as(mu)

        if init_noise is None:
            init_noise = torch.randn_like(mu)
        else:
            init_noise = init_noise.to(device=mu.device, dtype=mu.dtype)

        slots = mu + sigma * init_noise
        keys = self.to_k(inputs)
        values = self.to_v(inputs)
        final_attention = None

        for _ in range(self.iters):
            previous = slots
            queries = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bid,bjd->bij", queries, keys) * self.scale
            attention = logits.softmax(dim=1) + self.eps
            attention = attention / attention.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", values, attention)

            slots = self.gru(updates.reshape(-1, dim), previous.reshape(-1, dim))
            slots = slots.reshape(batch, self.num_slots, dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            final_attention = attention

        if return_attention:
            return slots, final_attention

        return slots


class SpatialBroadcastDecoder(nn.Module):
    """Decode spatially broadcast slots into RGB layers and soft object masks."""

    def __init__(
        self,
        slot_dim: int,
        hidden_dim: int = 64,
        out_channels: int = 3,
        resolution: int = 64,
    ):
        """Initialize the model layers."""
        super().__init__()
        self.resolution = resolution
        self.init_res = 8

        self.fc = nn.Linear(slot_dim, hidden_dim * self.init_res**2)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels + 1, 4, stride=2, padding=1),
        )

    def forward(self, slots):
        """Combine decoded slot layers using normalized pixel masks."""
        batch, num_slots, _ = slots.shape

        x = self.fc(slots).view(batch * num_slots, -1, self.init_res, self.init_res)
        x = self.dec(x).view(batch, num_slots, -1, self.resolution, self.resolution)

        rgb = x[:, :, :3]
        masks = x[:, :, 3:].softmax(dim=1)
        reconstruction = torch.sum(rgb * masks, dim=1)

        return reconstruction, masks


class SlotAttentionAE(nn.Module):
    """Reconstruct images using convolutional features, Slot Attention, and a decoder."""

    def __init__(self, num_slots: int = 4, slot_dim: int = 64, resolution: int = 64):
        """Initialize the model layers."""
        super().__init__()
        self.encoder = ConvEncoder(3, 64)
        self.project = nn.Linear(64, slot_dim)
        self.slot_attention = SlotAttention(num_slots, slot_dim, iters=3)
        self.decoder = SpatialBroadcastDecoder(slot_dim, 64, 3, resolution)

    def forward(self, x, init_noise=None, return_attention: bool = False):
        """Return reconstructions, slots, masks, and optional feature attention."""
        features = self.project(self.encoder(x))

        if return_attention:
            slots, attention = self.slot_attention(
                features, init_noise=init_noise, return_attention=True
            )
        else:
            slots = self.slot_attention(features, init_noise=init_noise)
            attention = None

        reconstruction, masks = self.decoder(slots)

        if return_attention:
            return reconstruction, slots, masks, attention

        return reconstruction, slots, masks


class FlatMLPProbFactorPredictor(nn.Module):
    """Predict factor means and log variances from concatenated slot vectors."""

    def __init__(
        self,
        num_slots: int = 4,
        slot_dim: int = 64,
        z_dim: int = 64,
        hidden_dim: int = 256,
        out_dim: int = 10,
        dropout: float = 0.0,
    ):
        """Initialize the model layers."""
        super().__init__()
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(num_slots * slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim * 2),
        )

    def forward(self, slots, return_latent: bool = False):
        """Predict factor distributions from an ordered slot concatenation."""
        flat = slots.reshape(slots.shape[0], -1)
        z = self.encoder(flat)
        output = self.head(z)

        mu = output[:, : self.out_dim]
        logvar = output[:, self.out_dim :]

        if return_latent:
            return mu, logvar, z, flat

        return mu, logvar


class DeepSetsProbFactorPredictor(nn.Module):
    """Predict factor distributions from pooled shared slot embeddings."""

    def __init__(
        self,
        slot_dim: int = 64,
        z_dim: int = 64,
        hidden_dim: int = 256,
        out_dim: int = 10,
        dropout: float = 0.0,
    ):
        """Initialize the model layers."""
        super().__init__()
        self.out_dim = out_dim

        self.phi = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim * 2),
        )

    def forward(self, slots, return_latent: bool = False):
        """Pool slot embeddings and predict factor means and log variances."""
        slot_features = self.phi(slots)
        set_representation = slot_features.mean(dim=1)
        z = self.rho(set_representation)
        output = self.head(z)

        mu = output[:, : self.out_dim]
        logvar = output[:, self.out_dim :]

        if return_latent:
            return mu, logvar, z, set_representation

        return mu, logvar
