"""
Étape 8 du pipeline ML : évaluation des modèles entraînés.

Au-delà des métriques statistiques classiques (MAE, RMSE, R²), la
Direction Accuracy (taux de bonne direction : le signe du rendement
prédit correspond-il au signe du rendement réel ?) est la métrique la
PLUS pertinente en finance — un modèle peut avoir un R² faible tout en
étant utile s'il devine correctement la direction plus souvent qu'au
hasard (cohérent avec metrics.py, qui calcule déjà cette même notion
pour le backtest global).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Proportion de prédictions dont le signe correspond à la réalité."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct))


def evaluate_model(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Calcule toutes les métriques d'évaluation pour un modèle déjà
    entraîné, sur le jeu de test (jamais vu pendant l'entraînement ni le
    tuning d'hyperparamètres).
    """
    y_pred = estimator.predict(X_test.values)

    return {
        "mae": mean_absolute_error(y_test.values, y_pred),
        "rmse": mean_squared_error(y_test.values, y_pred) ** 0.5,
        "r2": r2_score(y_test.values, y_pred),
        "direction_accuracy": direction_accuracy(y_test.values, y_pred),
        "predictions": y_pred,
    }


def feature_importance(estimator, feature_names: list, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> pd.Series:
    """
    Importance des features, extraite selon le type de modèle :
    coefficients (Ridge), feature_importances_ (Random Forest, Gradient
    Boosting, XGBoost, LightGBM), ou permutation importance en repli pour
    les modèles qui n'exposent ni l'un ni l'autre (Gaussian Process).
    """
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
    elif X_test is not None and y_test is not None:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            estimator, X_test.values, y_test.values, n_repeats=10, random_state=42, scoring="r2"
        )
        values = result.importances_mean
    else:
        raise ValueError(
            f"Impossible d'extraire l'importance des features pour {type(estimator).__name__} "
            "sans X_test/y_test (nécessaire pour la permutation importance)."
        )

    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def evaluate_all_models(trained: dict) -> dict:
    """
    Applique evaluate_model() à tous les modèles entraînés par
    trainer.train_all_models(), et ajoute l'importance des features de
    chacun.

    Args:
        trained: dict retourné par trainer.train_all_models()

    Returns:
        dict {nom_modèle: {..métriques.., 'feature_importance': Series}}
    """
    results = {}
    for name, info in trained["models"].items():
        metrics = evaluate_model(info["estimator"], trained["X_test"], trained["y_test"])
        metrics["feature_importance"] = feature_importance(
            info["estimator"], trained["selected_features"],
            X_test=trained["X_test"], y_test=trained["y_test"],
        )
        metrics["cv_score"] = info["cv_score"]
        metrics["params"] = info["params"]
        results[name] = metrics

    return results