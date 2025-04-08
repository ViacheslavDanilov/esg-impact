import logging
import os

import pandas as pd
import shap
from config import cfg
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src import ROOT_DIR
from src.data.scaler import DataScaler
from src.data.utils import generate_mock_data  # noqa

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    # Load the raw data
    df = pd.read_csv(cfg.train.data_path)
    df["IS_FINANCIAL_SECTOR"] = df["SECTOR"].apply(lambda x: 1 if "Financials" in x else 0)

    # Generate synthetic data - used for debugging purposes
    # df = generate_mock_data(df_raw, num_rows=500, seed=11)

    # Target variable (Y_params) = TOBIN_Q_RATIO
    Y = df[cfg.train.Y_params]

    # Treatment variable (T_params) = ESG scores
    T = df[cfg.train.T_params]

    # Control variables (W_params) = macro + fundamental
    W = df[cfg.train.W_params]

    # Variables for heterogeneity (X_params) = features that affect the strength of the ESG effect
    X = df[cfg.train.X_params]

    # Split the data into training and validation sets
    X_train, X_test, Y_train, Y_test, T_train, T_test, W_train, W_test = train_test_split(
        X,
        Y,
        T,
        W,
        test_size=0.2,
        random_state=cfg.train.seed,
        shuffle=True,
    )

    # Initialize the DataScaler
    scaler_path_w = os.path.join(ROOT_DIR, "data", "scaler_w.pkl")
    scaler_w = DataScaler(scaler_path_w)
    scaler_path_x = os.path.join(ROOT_DIR, "data", "scaler_x.pkl")
    scaler_x = DataScaler(scaler_path_x)

    # Fit and transform the training data
    W_train_scaled = scaler_w.fit_transform(W_train)
    W_test_scaled = scaler_w.transform(W_test)
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    # Inittialize the LinearDML model with RandomForestRegressor for model_y and model_t
    model = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=cfg.train.seed),
        model_t=RandomForestRegressor(n_estimators=100, random_state=cfg.train.seed),
        random_state=cfg.train.seed,
    )

    # Train the model on the training set
    model.fit(
        Y=Y_train,
        T=T_train,
        X=X_train_scaled,
        W=W_train_scaled,
    )

    ate = float(model.ate(X_test_scaled))  # Average Treatment Effect (ATE)
    cate = model.effect(X_test_scaled)  # Heterogeneous Treatment Effect (CATE)
    logger.info(f"Average Treatment Effect (ATE): {ate:.2f}")
    logger.info(f"Conditional Average Treatment Effect (CATE): {cate}")

    # Compute SHAP values for the model
    explainer = shap.Explainer(
        model.model_final,
    )  # FIXME: TypeError: The passed model is not callable and cannot be analyzed directly with the given masker! Model: StatsModelsLinearRegression(fit_intercept=False)
    shap_values = explainer(X_test_scaled)

    # Visualize SHAP values
    shap.summary_plot(shap_values, X_test_scaled)

    logger.info("Complete")


if __name__ == "__main__":
    main()
