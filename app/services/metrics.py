"""
Métriques de performance de portefeuille.

Centralise ici les fonctions de calcul de métriques utilisées à travers
le projet (ratio de Sharpe, Sortino, Max Drawdown...), pour respecter la
structure de fichiers prévue initialement (app/services/metrics.py).

Les fonctions de PERFORMANCE HISTORIQUE ex-ante (rendement/volatilité/
Sharpe d'un portefeuille à partir de poids donnés) vivent dans
markowitz.py, car elles sont utilisées comme briques de l'optimisation
elle-même (fonction objectif de scipy.optimize) — les dupliquer ici
créerait une source de vérité divergente entre l'optimisation et le
reporting.

Ce module regroupe donc spécifiquement les métriques de RISQUE et
d'ÉVALUATION HORS-ÉCHANTILLON (Sortino, Max Drawdown), utilisées par le
moteur de backtest (backtest.py) pour juger les méthodes sur des
données réellement réalisées, plutôt que sur des estimations ex-ante.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def compute_max_drawdown(daily_portfolio_returns: np.ndarray) -> float:
    """
    Perte maximale (en %) qu'aurait subie un investisseur entre un pic et
    le creux suivant, sur la période testée — la métrique de risque la
    plus parlante pour un investisseur non-technique ("combien pourrais-je
    perdre au pire moment ?").

        drawdown(t) = valeur(t) / max(valeur(0..t)) - 1
        max_drawdown = min(drawdown) sur toute la période
    """
    if len(daily_portfolio_returns) == 0:
        return 0.0
    wealth_curve = np.cumprod(1 + daily_portfolio_returns)
    running_max = np.maximum.accumulate(wealth_curve)
    drawdowns = wealth_curve / running_max - 1
    return float(drawdowns.min())


def compute_sortino_ratio(
    daily_portfolio_returns: np.ndarray, annualized_return: float, risk_free_rate: float
) -> float:
    """
    Variante du ratio de Sharpe qui ne pénalise que la volatilité À LA
    BAISSE (les mouvements positifs ne sont jamais considérés comme un
    "risque"). Plus pertinent que le Sharpe pour des investisseurs qui ne
    craignent pas la hausse, seulement la baisse.

        Sortino = (rendement - taux_sans_risque) / volatilité_baissière
    """
    downside_returns = daily_portfolio_returns[daily_portfolio_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_volatility = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_volatility == 0:
        return 0.0
    return float((annualized_return - risk_free_rate) / downside_volatility)


def compute_directional_accuracy(predicted_returns: np.ndarray, realized_returns: np.ndarray) -> float:
    """
    Taux de bonne direction : proportion des cas où le signe du rendement
    prédit correspond au signe du rendement réellement réalisé.

    Métrique complémentaire au R² : un modèle peut avoir un R² proche de
    0 (mauvaise précision sur la MAGNITUDE du rendement) tout en ayant un
    taux de bonne direction correct (bonne capacité à prédire au moins le
    SENS du mouvement, hausse ou baisse) — une information utile pour
    juger si le modèle a un intérêt pratique malgré un R² décevant.
    """
    if len(predicted_returns) == 0:
        return 0.0
    correct = np.sign(predicted_returns) == np.sign(realized_returns)
    return float(np.mean(correct))