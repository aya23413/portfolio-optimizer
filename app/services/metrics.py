"""
Calcul des indicateurs de performance d'un portefeuille :
rendement attendu, volatilité, ratio de Sharpe, etc.
"""

import numpy as np


def compute_performance(weights: dict, payload: dict, risk_free_rate: float = 0.02) -> dict:
    """
    Calcule les indicateurs de performance à partir des poids et des données de rendement.

    TODO:
        - récupérer la matrice de rendements/covariance depuis payload['returns_data']
        - rendement attendu = somme(poids_i * rendement_moyen_i)
        - volatilité = sqrt(w^T . Cov . w)
        - ratio de Sharpe = (rendement - taux_sans_risque) / volatilité
    """
    return {
        'expected_return': None,
        'volatility': None,
        'sharpe_ratio': None,
        'note': "Calcul à implémenter une fois les données de rendement branchées",
    }


def annualize_return(daily_return: float, periods: int = 252) -> float:
    """Annualise un rendement journalier."""
    return (1 + daily_return) ** periods - 1


def annualize_volatility(daily_volatility: float, periods: int = 252) -> float:
    """Annualise une volatilité journalière."""
    return daily_volatility * np.sqrt(periods)
