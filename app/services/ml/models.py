"""
Étape 5 du pipeline ML : définition des modèles candidats.

5 familles de modèles :
- Ridge
- Random Forest
- Gradient Boosting
- XGBoost
- Gaussian Process

Le pipeline compare les modèles et sélectionne celui qui obtient
les meilleures performances selon la validation temporelle.
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from xgboost import XGBRegressor


def create_ridge(random_state: int = 42) -> tuple:
    model = Ridge(random_state=random_state)

    param_distributions = {
        "alpha": [
            0.01,
            0.1,
            0.5,
            1.0,
            5.0,
            10.0,
            50.0,
        ],
    }

    return model, param_distributions


def create_rf(random_state: int = 42) -> tuple:
    model = RandomForestRegressor(
        random_state=random_state,
        n_jobs=1,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": ["sqrt", 0.5, 1.0],
    }

    return model, param_distributions


def create_gb(random_state: int = 42) -> tuple:
    model = GradientBoostingRegressor(
        random_state=random_state,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.85, 1.0],
    }

    return model, param_distributions


def create_xgboost(random_state: int = 42) -> tuple:
    model = XGBRegressor(
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
        objective="reg:squarederror",
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.0, 5.0],
    }

    return model, param_distributions


def create_gaussian_process(random_state: int = 42) -> tuple:

    kernel_rbf = (
        RBF(length_scale=1.0)
        + WhiteKernel(noise_level=1.0)
    )

    kernel_matern = (
        Matern(length_scale=1.0, nu=1.5)
        + WhiteKernel(noise_level=1.0)
    )

    model = GaussianProcessRegressor(
        random_state=random_state,
        normalize_y=True,
        n_restarts_optimizer=2,
    )

    param_distributions = {
        "kernel": [
            kernel_rbf,
            kernel_matern,
        ],
        "alpha": [
            1e-10,
            1e-5,
            1e-2,
        ],
    }

    return model, param_distributions


# ============================================================
# Registre central des modèles
# ============================================================

AVAILABLE_MODELS = {
    "ridge": create_ridge,
    "random_forest": create_rf,
    "gradient_boosting": create_gb,
    "xgboost": create_xgboost,
    "gaussian_process": create_gaussian_process,
}