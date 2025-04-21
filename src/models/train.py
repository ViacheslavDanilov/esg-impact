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
    features: list[str] = None,
    save_dir: str = os.path.join(ROOT_DIR, "eval"),
) -> None:
    tree_interpreter = SingleTreeCateInterpreter(
        include_model_uncertainty=True,
        max_depth=2,
        min_samples_leaf=10,
    )
    tree_interpreter.interpret(model, X)

    fig, ax = plt.subplots(figsize=(60, 20))
    tree_interpreter.plot(
        feature_names=features,
        precision=1,
        fontsize=16,
        ax=ax,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cate_tree.png")
    plt.savefig(save_path)
    plt.close()


def visualize_policy_tree(
    model: LinearDML,
    X: pd.DataFrame,
    features: list[str] = None,
    save_dir: str = os.path.join(ROOT_DIR, "eval"),
) -> None:
    policy_interpreter = SingleTreePolicyInterpreter(
        risk_level=None,
        max_depth=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.001,
    )
    policy_interpreter.interpret(model, X, sample_treatment_costs=0.02)

    fig, ax = plt.subplots(figsize=(40, 20))
    policy_interpreter.plot(
        feature_names=features,
        precision=1,
        fontsize=18,
        ax=ax,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "policy_tree.png")
    plt.savefig(save_path)
    plt.close()


def explain_shap(
    model: LinearDML,
    X: pd.DataFrame,
    outcome: str = "TOBIN_Q_RATIO",
    treatments: list[str] = None,
    save_dir: str = os.path.join(ROOT_DIR, "eval", "shap_summaries"),
):
    if treatments is None:
        treatments = list(model.shap_values(X)[outcome].keys())  # Auto-detect all treatments

    os.makedirs(save_dir, exist_ok=True)
    shap_values = model.shap_values(X)

    for treatment in treatments:
        values = shap_values[outcome][treatment]

        # Local force plot (optional, only for the first treatment/index)
        if treatment == treatments[0]:
            ind = 1
            shap.plots.force(values[ind], matplotlib=True)

        # Global summary plot
        plt.figure(figsize=(16, 10))
        shap.summary_plot(values, X, show=False)
        save_path = os.path.join(save_dir, f"{treatment}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved SHAP summary plot for '{treatment}' to: {save_path}")


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
        features=cfg.train.X_params,
    )

    # Visualize the policy tree
    visualize_policy_tree(
        model=model,
        X=data["X_test"],
        features=cfg.train.X_params,
    )

    # Explain the model using SHAP
    explain_shap(
        model=model,
        X=data["X_test"],
        outcome=cfg.train.Y_params[0],
        treatments=cfg.train.T_params,
    )

    logger.info("Complete")


if __name__ == "__main__":
    main()
