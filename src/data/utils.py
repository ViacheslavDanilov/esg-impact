import os

import numpy as np
import pandas as pd

from src import ROOT_DIR


def generate_mock_data(
    df: pd.DataFrame,
    num_rows: int,
    seed: int = 11,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    dummy_data = {}
    for column in df.columns:
        if df[column].dtype == "object":
            unique_values = df[column].dropna().unique()
            dummy_data[column] = np.random.choice(unique_values, num_rows)
        else:
            mean = df[column].mean()
            std = df[column].std()
            dummy_data[column] = np.random.normal(mean, std, num_rows)

    df_out = pd.DataFrame(dummy_data)
    df_out["YEAR"] = df_out["YEAR"].astype(int)
    return df_out


if __name__ == "__main__":
    data_path = os.path.join(ROOT_DIR, "data/raw.csv")
    df = pd.read_csv(data_path)
    df_mock = generate_mock_data(df, num_rows=100, seed=11)
    print(df_mock.head(5))
    print("Complete")
