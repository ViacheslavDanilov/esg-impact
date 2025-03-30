import os

import numpy as np
import pandas as pd

from src import ROOT_DIR


def generate_mock_data(
    df: pd.DataFrame,
    num_rows: int,
) -> pd.DataFrame:
    dummy_data = {}
    for column in df.columns:
        if df[column].dtype == "object":
            # For categorical columns, sample from the unique values
            unique_values = df[column].dropna().unique()
            dummy_data[column] = np.random.choice(unique_values, num_rows)
        else:
            # For numerical columns, use the mean and std to generate data
            mean = df[column].mean()
            std = df[column].std()
            dummy_data[column] = np.random.normal(mean, std, num_rows)

    df_out = pd.DataFrame(dummy_data)
    df_out["YEAR"] = df_out["YEAR"].astype(int)
    return df_out


if __name__ == "__main__":
    data_path = os.path.join(ROOT_DIR, "data/raw.csv")
    df = pd.read_csv(data_path)
    df_mock = generate_mock_data(df, 100)
    print(df_mock.head(5))
    print("Complete")
