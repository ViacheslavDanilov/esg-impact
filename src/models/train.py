import logging
import os

import pandas as pd
import shap
from config import cfg
from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter
from econml.dml import LinearDML
from matplotlib import pyplot as plt
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


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["IS_FINANCIAL_SECTOR"] = df["SECTOR"].apply(lambda x: 1 if "Financials" in x else 0)
    return df


def split_and_scale_data(df):
    Y = df[cfg.train.Y_params]
    T = df[cfg.train.T_params]
    W = df[cfg.train.W_params]
    X = df[cfg.train.X_params]

    X_train, X_test, Y_train, Y_test, T_train, T_test, W_train, W_test = train_test_split(
        X,
        Y,
        T,
        W,
        test_size=0.2,
        random_state=cfg.train.seed,
        shuffle=True,
    )

    scaler_path_w = os.path.join(ROOT_DIR, "data", "scaler_w.pkl")
    scaler_path_x = os.path.join(ROOT_DIR, "data", "scaler_x.pkl")
    scaler_w = DataScaler(scaler_path_w)
    scaler_x = DataScaler(scaler_path_x)

    W_train_scaled = scaler_w.fit_transform(W_train)
    W_test_scaled = scaler_w.transform(W_test)
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "T_train": T_train,
        "T_test": T_test,
        "W_train": W_train_scaled,
        "W_test": W_test_scaled,
    }


def train_model(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    T_train: pd.DataFrame,
    W_train: pd.DataFrame,
) -> LinearDML:
    model = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=cfg.train.seed),
        model_t=RandomForestRegressor(n_estimators=100, random_state=cfg.train.seed),
        random_state=cfg.train.seed,
    )
    model.fit(Y=Y_train, T=T_train, X=X_train, W=W_train)
    return model


def visualize_cate_tree(
    model: LinearDML,
    X: pd.DataFrame,
) -> None:
    tree_interpreter = SingleTreeCateInterpreter(
        include_model_uncertainty=True,
        max_depth=2,
        min_samples_leaf=10,
    )
    tree_interpreter.interpret(model, X)

    fig, ax = plt.subplots(figsize=(20, 10))
    tree_interpreter.plot(
        feature_names=[
            "FUND_BS_TOT_ASSET",
            "FUND_RETURN_ON_ASSET",
            "ESG_SCORE",
        ],
        precision=2,
        fontsize=12,
        ax=ax,
    )
    plt.tight_layout()
    save_path = os.path.join(ROOT_DIR, "eval", "cate_tree.png")
    plt.savefig(save_path)
    plt.close()


def visualize_policy_tree(
    model: LinearDML,
    X: pd.DataFrame,
) -> None:
    policy_interpreter = SingleTreePolicyInterpreter(
        risk_level=None,
        max_depth=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.001,
    )
    policy_interpreter.interpret(model, X, sample_treatment_costs=0.02)

    fig, ax = plt.subplots(figsize=(20, 10))
    policy_interpreter.plot(
        feature_names=[
            "FUND_BS_TOT_ASSET",
            "FUND_RETURN_ON_ASSET",
            "ESG_SCORE",
        ],
        precision=2,
        fontsize=12,
        ax=ax,
    )
    plt.tight_layout()
    save_path = os.path.join(ROOT_DIR, "eval", "policy_tree.png")
    plt.savefig(save_path)
    plt.close()


def explain_shap(
    model: LinearDML,
    X: pd.DataFrame,
):
    shap_values = model.shap_values(X)
    ind = 1
    shap.plots.force(
        shap_values["TOBIN_Q_RATIO"]["E_SUSTAINABLE_PRODUCT_ISSUE_SCORE"][ind],
        matplotlib=True,
    )
    shap.summary_plot(shap_values["TOBIN_Q_RATIO"]["E_SUSTAINABLE_PRODUCT_ISSUE_SCORE"])


def main():
    # Load and prepare the data
    df = load_and_prepare_data(cfg.train.data_path)
    data = split_and_scale_data(df)

    # Train the model
    model = train_model(
        X_train=data["X_train"],
        Y_train=data["Y_train"],
        T_train=data["T_train"],
        W_train=data["W_train"],
    )
    print(model.summary())

    # Visualize the CATE tree
    visualize_cate_tree(
        model=model,
        X=data["X_test"],
    )

    # Visualize the policy tree
    visualize_policy_tree(
        model=model,
        X=data["X_test"],
    )

    # Explain the model using SHAP
    explain_shap(
        model=model,
        X=data["X_test"],
    )

    logger.info("Complete")


if __name__ == "__main__":
    main()
