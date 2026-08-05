"""
Étape 7 du pipeline ML : entraînement de tous les modèles candidats.

Pipeline complet pour UN actif :
    X, y (preprocessing.prepare_dataset)
        -> split train/test temporel (dernier bloc = test, jamais mélangé)
        -> scaling (fit sur train seul)
        -> feature_selection (fit sur train seul, même logique anti-fuite)
        -> pour chaque modèle de models.AVAILABLE_MODELS :
               hyperparameter_tuning.tune_model()
        -> retourne tous les modèles entraînés, prêts pour evaluator.py
"""

import pandas as pd

from app.services.ml.models import AVAILABLE_MODELS
from app.services.ml.hyperparameter_tuning import tune_model
from app.services.ml.preprocessing import fit_scaler, apply_scaler
from app.services.ml.feature_selection import select_top_features


def temporal_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
    """
    Split train/test respectant l'ordre chronologique (le test set est
    TOUJOURS la période la plus récente) — jamais de split aléatoire sur
    des séries temporelles financières.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test



# Sous-ensemble de modèles utilisé en fast_mode (backtest à fenêtres
# glissantes : le pipeline complet tourne une fois PAR FENÊTRE, potentiel-
# lement 3-5 fois pour un seul clic utilisateur -> il faut un compromis
# vitesse/précision différent de l'appel "normal" du dashboard, qui ne
# tourne qu'une fois. On garde Ridge + Gradient Boosting : un modèle
# linéaire rapide et un modèle non-linéaire parmi les plus performants
# habituellement, on écarte XGBoost/LightGBM/Gaussian Process/Random
# Forest (redondants avec Gradient Boosting sur ce cas d'usage, et plus
# coûteux à entraîner/tuner).
FAST_MODE_MODELS = ["ridge", "gradient_boosting"]


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    top_k_features: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
    fast_mode: bool = False,
) -> dict:
    """
    Entraîne les modèles de models.AVAILABLE_MODELS sur un actif donné.

    Args:
        fast_mode: si True, entraîne seulement un sous-ensemble réduit de
                   modèles (FAST_MODE_MODELS) avec un tuning minimal (voir
                   hyperparameter_tuning.tune_model). Destiné aux appels
                   répétés à forte volumétrie (backtest à fenêtres
                   glissantes), où le pipeline complet à 6 modèles
                   tournerait une fois par fenêtre -> coût multiplié par
                   le nombre de fenêtres, inadapté à un usage interactif.

    Returns:
        dict {
            'X_train', 'X_test', 'y_train', 'y_test': données utilisées
                (features déjà réduites + standardisées, pour evaluator.py)
            'selected_features': liste des features retenues
            'scaler': StandardScaler ajusté (réutilisable pour prédire
                      sur de nouvelles données, voir predictor.py)
            'models': dict {nom_modèle: {'estimator', 'params', 'cv_score'}}
        }
    """
    X_train_raw, X_test_raw, y_train, y_test = temporal_train_test_split(X, y, test_size)

    # Sélection de features : fit UNIQUEMENT sur le train, comme le scaler
    selected_features = select_top_features(X_train_raw, y_train, top_k=top_k_features)
    X_train_sel = X_train_raw[selected_features]
    X_test_sel = X_test_raw[selected_features]

    scaler = fit_scaler(X_train_sel)
    X_train_scaled = apply_scaler(scaler, X_train_sel)
    X_test_scaled = apply_scaler(scaler, X_test_sel)

    models_to_train = FAST_MODE_MODELS if fast_mode else list(AVAILABLE_MODELS.keys())
    tuning_kwargs = {"n_splits": 2, "n_iter": 4} if fast_mode else {}

    trained_models = {}
    for name in models_to_train:
        factory = AVAILABLE_MODELS[name]
        model, param_distributions = factory(random_state=random_state)
        best_estimator, best_params, best_score = tune_model(
            model, param_distributions, X_train_scaled.values, y_train.values,
            random_state=random_state, **tuning_kwargs,
        )
        trained_models[name] = {
            "estimator": best_estimator,
            "params": best_params,
            "cv_score": best_score,
        }

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "selected_features": selected_features,
        "scaler": scaler,
        "models": trained_models,
    }