import torch
import torch.nn as nn


class ShukuchiAutoencoder(nn.Module):
    """32‑k CpG → latent autoencoder used for feature extraction."""
    def __init__(self, in_size: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_size, 2048), nn.BatchNorm1d(2048), nn.PReLU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.PReLU(),
            nn.Linear(1024, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 1024), nn.BatchNorm1d(1024), nn.PReLU(),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.PReLU(),
            nn.Linear(2048, in_size), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class TabularTransformer(nn.Module):
    """Transformer‑based classifier for latent features."""
    def __init__(self, n_feats: int, n_cls: int, cfg):
        super().__init__()
        self.in_norm = nn.BatchNorm1d(n_feats)
        self.in_drop = nn.Dropout(cfg.dropout * 0.5)

        self.embed = nn.Sequential(
            nn.Linear(n_feats, cfg.d_model * 2), nn.GELU(),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.pos = nn.Parameter(torch.randn(1, n_feats, cfg.d_model) * 0.02)
        self.to_tok = nn.Linear(1, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_model * 4,
            cfg.dropout, activation="gelu", batch_first=True
        )
        self.tx = nn.TransformerEncoder(enc_layer, cfg.n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.cls = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(),
            nn.Dropout(cfg.dropout * 0.25),
            nn.Linear(cfg.d_model // 2, n_cls),
        )

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_drop(self.in_norm(x))
        glob = self.embed(x)
        tok = self.to_tok(x.unsqueeze(-1)) + self.pos
        pooled = self.pool(self.tx(tok).transpose(1, 2)).squeeze(-1)
        return self.cls(glob + pooled)
