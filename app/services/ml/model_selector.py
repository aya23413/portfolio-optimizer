"""
Étape 9 du pipeline ML : sélection automatique du meilleur modèle.

Contrairement à l'ancien ml_black_litterman.py (v3 en production), qui
choisissait le modèle sur le R² seul, ce module utilise un score composite
:

    score = 40% x Direction Accuracy + 30% x R² + 30% x Sharpe obtenu

Justification (approche gestion quantitative) : en finance, un modèle
avec un R² moyen mais qui devine bien la direction et produit un
portefeuille rentable peut être préférable à un modèle avec un R² plus
élevé mais peu exploitable en pratique. Le R² seul mesure un ajustement
statistique, pas une utilité économique.
"""

import numpy as np
import pandas as pd


def _min_max_normalize(values: dict) -> dict:
    """
    Normalise un dict {nom: valeur} entre 0 et 1, pour pouvoir combiner
    des métriques d'échelles différentes (R² peut être négatif, Sharpe
    peut être n'importe où, Direction Accuracy est déjà entre 0 et 1).
    Si toutes les valeurs sont égales, retourne 0.5 partout (évite une
    division par zéro sans favoriser arbitrairement un modèle).
    """
    arr = np.array(list(values.values()), dtype=float)
    low, high = arr.min(), arr.max()
    if high == low:
        return {k: 0.5 for k in values}
    return {k: (v - low) / (high - low) for k, v in values.items()}


def simple_strategy_sharpe(y_test: pd.Series, y_pred: np.ndarray, risk_free_daily: float = 0.0) -> float:
    """
    Proxy simplifié du "Sharpe obtenu" par modèle : simule une stratégie
    qui, à chaque prédiction, prend une position proportionnelle au signe
    et à l'amplitude du rendement prédit (long si positif, short si
    négatif, taille = amplitude bornée à [-1, 1]), puis calcule le ratio
    de Sharpe de cette série de rendements sur le jeu de test.

    NOTE : c'est un proxy, pas un vrai backtest multi-actifs avec
    Black-Litterman (trop coûteux à faire pour CHAQUE combinaison
    modèle x actif lors de la sélection) — le vrai backtest final (comparant
    Markowitz / Black-Litterman / IA) reste celui de backtest.py, sur le
    portefeuille complet.
    """
    position = np.clip(y_pred / (np.abs(y_pred).max() + 1e-9), -1, 1)
    strategy_returns = position * y_test.values

    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    if std_return == 0:
        return 0.0
    return float((mean_return - risk_free_daily) / std_return)


def select_best_model(
    evaluation_results: dict,
    y_test: pd.Series,
    weights: dict = None,
) -> tuple:
    """
    Choisit le meilleur modèle parmi ceux évalués par
    evaluator.evaluate_all_models(), selon le score composite.

    Args:
        evaluation_results: dict retourné par evaluate_all_models()
        y_test: cible réelle du jeu de test (pour calculer le Sharpe proxy)
        weights: pondération du score composite (défaut : 40/30/30)

    Returns:
        (nom_du_meilleur_modèle, dict des scores détaillés par modèle)
    """
    weights = weights or {"direction_accuracy": 0.4, "r2": 0.3, "sharpe": 0.3}

    sharpe_by_model = {
        name: simple_strategy_sharpe(y_test, result["predictions"])
        for name, result in evaluation_results.items()
    }
    direction_by_model = {name: r["direction_accuracy"] for name, r in evaluation_results.items()}
    r2_by_model = {name: r["r2"] for name, r in evaluation_results.items()}

    # Direction accuracy est déjà entre 0 et 1 (pas besoin de normaliser),
    # mais R² et Sharpe ont des échelles arbitraires -> normalisation
    # min-max entre les modèles candidats.
    r2_norm = _min_max_normalize(r2_by_model)
    sharpe_norm = _min_max_normalize(sharpe_by_model)

    composite_scores = {}
    for name in evaluation_results:
        composite_scores[name] = (
            weights["direction_accuracy"] * direction_by_model[name]
            + weights["r2"] * r2_norm[name]
            + weights["sharpe"] * sharpe_norm[name]
        )

    best_model_name = max(composite_scores, key=composite_scores.get)

    details = {
        name: {
            "composite_score": composite_scores[name],
            "direction_accuracy": direction_by_model[name],
            "r2": r2_by_model[name],
            "sharpe_proxy": sharpe_by_model[name],
        }
        for name in evaluation_results
    }

    return best_model_name, details