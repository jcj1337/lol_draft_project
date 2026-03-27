from __future__ import annotations

import torch
from torch import nn


class DraftTransformer(nn.Module):
    """
    Input:
        champ_ids:         [B, 10]
        team_ids:          [B, 10]   (0 for team A, 1 for team B)
        role_ids:          [B, 10]   (0=top, 1=jg, 2=mid, 3=adc, 4=sup)
        numeric_features:  [B, F]

    Output:
        logits:            [B]
    """

    def __init__(
        self,
        num_champions: int,
        num_numeric_features: int,
        embed_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 32,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim

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

        self.subclass_embedding = nn.Embedding(
            num_embeddings=8,
            embedding_dim=embed_dim,
        )

        self.scaling_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=embed_dim,
        )

        # project full numeric feature vector into one token
        self.numeric_token_proj = nn.Linear(num_numeric_features, embed_dim)

        # learned CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

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

        # no numeric concat anymore; head only sees CLS output
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        numeric_features: torch.Tensor,
        champ_ids: torch.Tensor,
        team_ids: torch.Tensor,
        role_ids: torch.Tensor,
        subclass_ids: torch.Tensor,
        scaling_ids: torch.Tensor,
    ) -> torch.Tensor:
        champ_emb = self.champion_embedding(champ_ids)       # [B, 10, D]
        team_emb = self.team_embedding(team_ids)             # [B, 10, D]
        role_emb = self.role_embedding(role_ids)             # [B, 10, D]
        subclass_emb = self.subclass_embedding(subclass_ids) # [B, 10, D]
        scaling_emb = self.scaling_embedding(scaling_ids)    # [B, 10, D]

        # 10 draft tokens
        x = champ_emb + team_emb + role_emb + subclass_emb + scaling_emb  # [B, 10, D]

        batch_size = x.size(0)

        # 1 CLS token
        cls_tokens = self.cls_token.expand(batch_size, 1, self.embed_dim)  # [B, 1, D]

        # 1 numeric token from the full numeric feature vector
        numeric_token = self.numeric_token_proj(numeric_features).unsqueeze(1)  # [B, 1, D]

        # sequence: CLS + 10 draft tokens + 1 numeric token
        x = torch.cat([cls_tokens, x, numeric_token], dim=1)  # [B, 12, D]

        x = self.encoder(x)

        # final CLS representation
        pooled = x[:, 0, :]  # [B, D]

        logits = self.head(pooled).squeeze(-1)
        return logits