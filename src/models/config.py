import os
from typing import List

from pydantic import BaseModel

from src import ROOT_DIR


class Training(BaseModel):
    """Configuration settings for model training."""

    data_path: str = os.path.join(ROOT_DIR, "data/processed.csv")
    Y_params: List[str] = ["TOBIN_Q_RATIO"]
    T_params: List[str] = [
        "E_SUSTAINABLE_PRODUCT_ISSUE_SCORE",
        "E_ENVIRON_AIR_QUALITY_ISSUE_SCORE",
        "E_CLIMATE_EXPOS_ISSUE_SCORE",
        "E_ECOLOGICAL_IMPACT_ISSUE_SCORE",
        "E_ENERGY_MANAGEMENT_ISSUE_SCORE",
        "E_GHG_EMISSIONS_MNGT_ISSUE_SCORE",
        "E_WATER_MANAGEMENT_ISSUE_SCORE",
        "E_WASTE_MANAGEMENT_ISSUE_SCORE",
        "S_SOCIAL_SPPL_CHN_MGMT_ISSUE_SCORE",
        "S_LABOR_EMPLYMNT_PRACT_ISSUE_SCORE",
        "S_PRODUCT_QUALITY_MNGT_ISSUE_SCORE",
        "S_DATA_SEC_CSTMR_PRIVCY_ISSUE_SCR",
        "S_COMM_RIGHTS_RELATION_ISSUE_SCORE",
        "G_BOARD_COMPOSITION_SCORE",
        "G_EXECUTIVE_COMPENSATION_SCORE",
        "G_SHAREHOLDER_RIGHTS_THEME_SCR",
        "G_AUDIT_THEME_SCORE",
    ]
    W_params: List[str] = [
        "IS_FINANCIAL_SECTOR",
        "FUND_CRNCY_ADJ_MKT_CAP",
        "FUND_TOT_DEBT_TO_TOT_EQY",
        # "FUND_BS_TOT_ASSET",
        "FUND_NET_DEBT_EBITDA_ADJUSTED",
        # "FUND_RETURN_ON_ASSET",
        "FUND_RETURN_COM_EQY",
        "MACRO_INFLATION",
        "MACRO_UNEMPLOYMENT",
        "MACRO_GDP_GROWTH",
        "MACRO_PMI",
    ]
    X_params: List[str] = [
        "FUND_BS_TOT_ASSET",  # Размер компании
        "FUND_RETURN_ON_ASSET",  # Доходность
        "ESG_SCORE",  # Интегральный скор, может быть полезен как доп. индикатор
    ]
    seed: int = 11
    save_dir: str = os.path.join(ROOT_DIR, "models")


class Settings(BaseModel):
    """Main configuration class."""

    train: Training = Training()


cfg = Settings()
