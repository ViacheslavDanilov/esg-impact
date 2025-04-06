import logging

import pandas as pd
import shap
from config import cfg
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.data.utils import generate_mock_data

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
    # Загрузка исходных данных
    df_raw = pd.read_csv(cfg.train.data_path)

    # Генерация синтетических данных
    df = generate_mock_data(df_raw, num_rows=500, seed=11)

    # Зависимая переменная (Y_params) = TOBIN_Q_RATIO
    Y = df[cfg.train.Y_params]

    # Вмешательство (T_params) = ESG-показатели
    T = df[cfg.train.T_params]

    # Контрольные переменные (W_params) = макро + фундаментальные
    W = df[cfg.train.W_params]

    # Переменные для гетерогенности (X_params) - Признаки, от которых зависит сила эффекта ESG
    # Если мы хотим оценить, как влияние ESG на Q Tobin варьируется в зависимости от:
    # Размера компании (CRNCY_ADJ_MKT_CAP),
    # Сектора (FINANCIAL_DUMMY),
    # Года наблюдения (Year).
    # Если гетерогенность не требуется, X_params=None или константа.
    # TODO: define X_params in order to use the model
    X = df[cfg.train.X_params] if cfg.train.X_params else pd.DataFrame(index=df.index)

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_test, Y_train, Y_test, T_train, T_test, W_train, W_test = train_test_split(
        X,
        Y,
        T,
        W,
        test_size=0.2,
        random_state=cfg.train.seed,
        shuffle=True,
    )

    # Инициализация модели LinearDML с RandomForestRegressor для model_y и model_t
    model = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=cfg.train.seed),
        model_t=RandomForestRegressor(n_estimators=100, random_state=cfg.train.seed),
        random_state=cfg.train.seed,
    )

    # Обучение модели на обучающей выборке
    model.fit(
        Y=Y_train,
        T=T_train,
        X=X_train,  # TODO: define X_params in order to use the model
        W=W_train,
    )

    ate = model.ate(X)  # Средний эффект ESG
    cate = model.effect(X)  # Гетерогенные эффекты
    logger.info(f"Average Treatment Effect (ATE): {ate}")
    logger.info(f"Conditional Average Treatment Effect (CATE): {cate}")

    # Получение SHAP-значений для модели
    explainer = shap.Explainer(model.model_final)
    shap_values = explainer(X_test)

    # Визуализация SHAP-значений
    shap.summary_plot(shap_values, X_test)

    logger.info("Complete")


if __name__ == "__main__":
    main()
