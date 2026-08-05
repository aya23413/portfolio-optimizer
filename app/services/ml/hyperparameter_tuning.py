"""
Étape 6 du pipeline ML : recherche d'hyperparamètres.

RandomizedSearchCV plutôt que GridSearchCV exhaustif : avec la taille du
dataset ici (quelques centaines d'observations par actif), une grille
complète est inutilement coûteuse pour un gain marginal. TimeSeriesSplit
est OBLIGATOIRE (jamais de K-Fold classique) : un split aléatoire
classique entraînerait le modèle sur des dates futures pour prédire des
dates passées -> fuite temporelle, cohérent avec la règle déjà appliquée
dans l'ancien ml_black_litterman.py.
"""

import warnings

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def tune_model(
    model,
    param_distributions: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    n_iter: int = 10,
    random_state: int = 42,
) -> tuple:
    """
    Recherche les meilleurs hyperparamètres pour un modèle donné, par
    validation croisée temporelle.

    Args:
        model: instance de modèle non entraînée (sklearn)
        param_distributions: espace de recherche (voir models.py)
        X, y: données d'entraînement (déjà standardisées si nécessaire)
        n_splits: nombre de découpes TimeSeriesSplit
        n_iter: nombre de combinaisons testées (RandomizedSearchCV)

    Returns:
        (best_estimator, best_params, best_score) — best_score est le R²
        moyen sur les folds de validation.
    """
    cv = TimeSeriesSplit(n_splits=n_splits)

    # n_iter ne peut pas dépasser le nombre de combinaisons possibles
    n_combinations = 1
    for values in param_distributions.values():
        n_combinations *= len(values)
    effective_n_iter = min(n_iter, n_combinations)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=effective_n_iter,
        cv=cv,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    with warnings.catch_warnings():
        # Certains folds initiaux (peu d'observations) peuvent produire
        # des warnings de convergence sans conséquence sur le résultat
        # final -> on les tait ici plutôt que de polluer les logs.
        warnings.simplefilter("ignore")
        search.fit(X, y)

    return search.best_estimator_, search.best_params_, search.best_score_