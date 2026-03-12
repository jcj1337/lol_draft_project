from __future__ import annotations

import torch
from torch import nn

# basic rn
# logic flow: 
# 10 tokens each vector of dim 64 [10, 64] >
# Attention layer, now vectors are "context-aware" > 
# feed-forward net for each token, dim 64 > 128 > 64 (helps process what the attention actually means) >
# pool all processed vectors into one draft vector >
# ANN on the vector for logit >
# output probability

class DraftTransformer(nn.Module):
    """
    Input:
        champ_ids: [B, 10]
        team_ids:  [B, 10]   (0 for team A, 1 for team B)

    Output:
        logits:    [B]
    """

    def __init__(
        # embedded dim 64, 4 heads for now, 2 layers 
        # dropout for the love ofthe game
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

        # Champion id embedding
        self.champion_embedding = nn.Embedding(
            num_embeddings=num_champions,
            embedding_dim=embed_dim,
        )

        # Team embedding: 0 = team A, 1 = team B (for synergies/counters)
        self.team_embedding = nn.Embedding(
            num_embeddings=2,
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

    def forward(self, champ_ids: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        """
        champ_ids: [B, 10]
        team_ids:  [B, 10]
        returns logits: [B]
        """
        champ_emb = self.champion_embedding(champ_ids)   # [B, 10, D]
        team_emb = self.team_embedding(team_ids)         # [B, 10, D]

        x = champ_emb + team_emb                         # [B, 10, D]
        x = self.encoder(x)                              # [B, 10, D]

        pooled = x.mean(dim=1)                           # [B, D]
        logits = self.head(pooled).squeeze(-1)           # [B]

        return logits