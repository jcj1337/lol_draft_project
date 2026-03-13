from __future__ import annotations

import torch
from torch import nn


class DraftTransformer(nn.Module):
    """
    Input:
        champ_ids: [B, 10]
        team_ids:  [B, 10]   (0 for team A, 1 for team B)
        role_ids:  [B, 10]   (0=top, 1=jg, 2=mid, 3=adc, 4=sup)

    Output:
        logits:    [B]
    """

    def __init__(
        self,
        num_champions: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 64,
    ):
        super().__init__()

        self.champion_embedding = nn.Embedding(
            num_embeddings=num_champions,
            embedding_dim=embed_dim,
        )

        self.team_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embed_dim,
        )

        self.role_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=embed_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        champ_ids: torch.Tensor,
        team_ids: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        champ_emb = self.champion_embedding(champ_ids)   # [B, 10, D]
        team_emb = self.team_embedding(team_ids)         # [B, 10, D]
        role_emb = self.role_embedding(role_ids)         # [B, 10, D]

        x = champ_emb + team_emb + role_emb
        x = self.encoder(x)

        pooled = x.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)

        return logits