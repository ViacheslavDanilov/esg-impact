import logging
import os
from typing import Optional

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import ROOT_DIR


class DataScaler:
    """A class for scaling data."""

    def __init__(self, scaler_path: str = "scaler.pkl") -> None:
        """Initializes the DataScaler with an optional path to save/load the scaler.

        Parameters:
        - scaler_path (str): Path to save or load the scaler object.
        """
        self.scaler: Optional[StandardScaler] = None
        self.scaler_path = scaler_path
        self.logger = logging.getLogger(__name__)

    def fit(self, data: pd.DataFrame) -> None:
        """Fits the StandardScaler to the data and saves the scaler to disk.

        Parameters:
        - data (pd.DataFrame): Training data to fit the scaler.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        joblib.dump(self.scaler, self.scaler_path)
        self.logger.info(f"Scaler fitted and saved to {self.scaler_path}.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the fitted scaler.

        Parameters:
        - data (pd.DataFrame): Data to be transformed.

        Returns:
        - Transformed data as a NumPy array.
        """
        if self.scaler is None:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info(f"Scaler loaded from {self.scaler_path}.")
            else:
                raise FileNotFoundError("Scaler has not been fitted or saved.")
        return self.scaler.transform(data)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits the scaler to the data, transforms it, and saves the scaler to disk.

        Parameters:
        - data (pd.DataFrame): Data to fit and transform.

        Returns:
        - Transformed data as a NumPy array.
        """
        self.fit(data)
        return self.transform(data)


if "__main__" == __name__:
    from sklearn.model_selection import train_test_split

    # Sample data
    data = pd.DataFrame(
        {
            "feature1": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
            "feature2": [5, 15, 25, 35, 45, 10, 20, 30, 40, 50],
            "feature3": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            "feature4": [-6, 37, 863, 375, -209, 63, 28, 0, -1, 2],
        },
    )

    # Splitting data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=11)

    # Initialize the DataScaler
    scaler_path = os.path.join(ROOT_DIR, "data", "scaler.pkl")
    scaler = DataScaler(scaler_path)

    # Fit and transform the training data
    train_data_normalized = scaler.fit_transform(train_data)

    # Transform the test data
    test_data_normalized = scaler.transform(test_data)

    print("Normalized Training Data:\n", train_data_normalized)
    print("Normalized Test Data:\n", test_data_normalized)
