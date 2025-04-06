import os
from typing import List, Optional

from pydantic import BaseModel

from src import ROOT_DIR


class Training(BaseModel):
    """Configuration settings for model training."""

    data_path: str = os.path.join(ROOT_DIR, "data/raw.csv")
    Y_params: List[str] = ["TOBIN_Q_RATIO"]
    T_params: List[str] = ["ESG_SCORE"]
    W_params: List[str] = [
        "FUND_CRNCY_ADJ_MKT_CAP",
        "FUND_TOT_DEBT_TO_TOT_EQY",
        "FUND_BS_TOT_ASSET",
        "FUND_NET_DEBT_EBITDA_ADJUSTED",
        "FUND_RETURN_ON_ASSET",
        "FUND_RETURN_COM_EQY",
        "MACRO_INFLATION",
        "MACRO_UNEMPLOYMENT",
        "MACRO_GDP_GROWTH",
        "MACRO_PMI",
    ]
    X_params: Optional[List[str]] = None
    seed: int = 11
    save_dir: str = os.path.join(ROOT_DIR, "models")


class Settings(BaseModel):
    """Main configuration class."""

    train: Training = Training()


cfg = Settings()
