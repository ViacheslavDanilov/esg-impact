import pandas as pd
from config import cfg
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor

from src.data.utils import generate_mock_data


def main():
    # Create mock data
    df_raw = pd.read_csv(cfg.train.data_path)
    df = generate_mock_data(df_raw, num_rows=500, seed=11)

    # Data preparation
    # Зависимая переменная (Y_params) = TOBIN_Q_RATIO
    Y = df[cfg.train.Y_params]

    # Вмешательство (T_params) = ESG-показатели
    # Вариант 1: Использовать ESG_SCORE как общий показатель
    # Вариант 2: Разделить на E, S, G и оценить их влияние по отдельности (требует больше данных и проверки на мультиколлинеарность)
    T = df[cfg.train.T_params]

    # Контрольные переменные (W_params) = макро + фундаментальные
    W = df[cfg.train.W_params]

    # Переменные для гетерогенности (X_params) - Признаки, от которых зависит сила эффекта ESG
    # Если мы хотитм оценить, как влияние ESG на Q Tobin варьируется в зависимости от:
    # Размера компании (CRNCY_ADJ_MKT_CAP),
    # Сектора (FINANCIAL_DUMMY),
    # Года наблюдения (Year).
    # Если гетерогенность не требуется, X_params=None или константа.
    X = df[cfg.train.X_params] if cfg.train.X_params else None

    # Training a DML Model
    model = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        random_state=cfg.train.seed,
    )
    model.fit(
        Y=Y,
        T=T,
        X=X,
        W=W,
    )
    ate = model.ate(X)  # Средний эффект ESG
    cate = model.effect(X)  # Гетерогенные эффекты
    print(f"Average Treatment Effect (ATE): {ate}")
    print(f"Conditional Average Treatment Effect (CATE): {cate}")
    print("Complete")


if __name__ == "__main__":
    main()
