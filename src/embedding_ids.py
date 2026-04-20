from __future__ import annotations

from cProfile import label
from cProfile import label
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#------------------------------ categorical ------------------------------
TEAM_A_COLS = ["team_a_top", "team_a_jg", "team_a_mid", "team_a_adc", "team_a_sup"]
TEAM_B_COLS = ["team_b_top", "team_b_jg", "team_b_mid", "team_b_adc", "team_b_sup"]

TEAM_A_SUBCLASS_COLS = [
    "team_a_top_subclass",
    "team_a_jg_subclass",
    "team_a_mid_subclass",
    "team_a_adc_subclass",
    "team_a_sup_subclass",
]
TEAM_B_SUBCLASS_COLS = [
    "team_b_top_subclass",
    "team_b_jg_subclass",
    "team_b_mid_subclass",
    "team_b_adc_subclass",
    "team_b_sup_subclass",
]

TEAM_A_SCALING_COLS = [
    "team_a_top_scaling_type",
    "team_a_jg_scaling_type",
    "team_a_mid_scaling_type",
    "team_a_adc_scaling_type",
    "team_a_sup_scaling_type",
]
TEAM_B_SCALING_COLS = [
    "team_b_top_scaling_type",
    "team_b_jg_scaling_type",
    "team_b_mid_scaling_type",
    "team_b_adc_scaling_type",
    "team_b_sup_scaling_type",
]

LABEL_COL = "team_a_win"

#------------------------------ numeric ------------------------------
ROLE_IDS = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

SUBCLASS_TO_ID = {
    "tank": 0,
    "bruiser": 1,
    "mage": 2,
    "marksman": 3,
    "assassin": 4,
    "enchanter": 5,
    "engage": 6,
    "battlemage": 7,
}

SCALING_TO_ID = {
    "early": 0,
    "mid": 1,
    "late": 2,
}

NUMERIC_FEATURE_COLS = [
    # lane win-rate signal
    "top_wr_diff",
    "jg_wr_diff",
    "mid_wr_diff",
    "adc_wr_diff",
    "sup_wr_diff",

    # confidence / sample-size signal
    "top_low_games_flag",
    "jg_low_games_flag",
    "mid_low_games_flag",
    "adc_low_games_flag",
    "sup_low_games_flag",
    "low_games_count",
    "avg_games_diff",

    # subclass composition
    "team_a_num_tanks",
    "team_a_num_battlemages",
    "team_a_num_bruisers",
    "team_a_num_mages",
    "team_a_num_marksmen",
    "team_a_num_assassins",
    "team_a_num_enchanters",
    "team_a_num_engages",

    "team_b_num_tanks",
    "team_b_num_battlemages",
    "team_b_num_bruisers",
    "team_b_num_mages",
    "team_b_num_marksmen",
    "team_b_num_assassins",
    "team_b_num_enchanters",
    "team_b_num_engages",

    # comp shape / weird role flags
    "team_a_frontline_count",
    "team_b_frontline_count",
    "team_a_top_is_enchanter",
    "team_b_top_is_enchanter",
    "team_a_jg_is_enchanter",
    "team_b_jg_is_enchanter",
]
ALL_CHAMP_COLS = TEAM_A_COLS + TEAM_B_COLS
LABEL_COL = "team_a_win"
ROLE_IDS = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

def load_cleaned_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def build_champion_ids(df: pd.DataFrame) -> Dict[str, int]:
    """
    Map champion name to integer id, e.g., aatrox > 0, ahri > 1, etc.
    """
    champs = sorted({str(champ).strip() for col in ALL_CHAMP_COLS for champ in df[col].tolist()})
    champ_to_id = {champ: idx for idx, champ in enumerate(champs)}
    return champ_to_id


class DraftDataset(Dataset):
    """
    Returns:
        champ_ids: shape [10]
        team_ids:  shape [10]  where first 5 are team A and last 5 are team B
        label: scalar 0/1
    """

    def __init__(self, df: pd.DataFrame, champ_to_id: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.champ_to_id = champ_to_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        team_a = [str(row[col]).strip() for col in TEAM_A_COLS]
        team_b = [str(row[col]).strip() for col in TEAM_B_COLS]

        champ_ids = torch.tensor(
            [self.champ_to_id[c] for c in team_a + team_b],
            dtype=torch.long,
        )

        team_ids = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
        role_ids = torch.tensor(ROLE_IDS, dtype=torch.long)

        # aggregate
        subclass_cols = TEAM_A_SUBCLASS_COLS + TEAM_B_SUBCLASS_COLS
        scaling_cols = TEAM_A_SCALING_COLS + TEAM_B_SCALING_COLS

        subclass_ids = torch.tensor(
            [SUBCLASS_TO_ID[str(row[col]).strip()] for col in subclass_cols],
            dtype=torch.long,
        )

        scaling_ids = torch.tensor(
            [SCALING_TO_ID[str(row[col]).strip()] for col in scaling_cols],
            dtype=torch.long,
        )

        numeric_features = torch.tensor(
            [float(row[col]) for col in NUMERIC_FEATURE_COLS],
            dtype=torch.float32,
        )

        label = torch.tensor(float(row[LABEL_COL]), dtype=torch.float32)

        return {
            "champ_ids": champ_ids,
            "team_ids": team_ids,
            "role_ids": role_ids,
            "subclass_ids": subclass_ids,
            "scaling_ids": scaling_ids,
            "numeric_features": numeric_features,
            "label": label,
        }


class DraftEmbeddingInput(nn.Module):
    """
    Turns champion ids into trainable embedding vectors

    Output shape:
        [batch_size, 10, embed_dim]
    """

    def __init__(self, num_champions: int, embed_dim: int):
        super().__init__()
        self.champion_embedding = nn.Embedding(num_embeddings=num_champions, embedding_dim=embed_dim)
        self.team_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)

    def forward(
        self,
        numeric_features: torch.Tensor,
        champ_ids: torch.Tensor,
        team_ids: torch.Tensor,
        role_ids: torch.Tensor,
        subclass_ids: torch.Tensor,
        scaling_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        champ_ids: [B, 10]
        team_ids:  [B, 10]
        """
        champ_emb = self.champion_embedding(champ_ids)  # [B, 10, D]
        team_emb = self.team_embedding(team_ids)        # [B, 10, D]

        # Combine champion identity + which team they belong to (to model synergies + counters)
        x = champ_emb + team_emb
        return x
