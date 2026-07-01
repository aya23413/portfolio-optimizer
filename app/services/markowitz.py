"""
Optimisation de portefeuille selon le modèle de Markowitz (Moyenne-Variance).

Approche classique : minimiser la volatilité pour un rendement cible,
ou maximiser le ratio de Sharpe.
"""

import numpy as np


def optimize(payload: dict) -> dict:
    """
    Calcule les poids optimaux selon Markowitz.

    payload attendu :
        {
            "tickers": ["AAPL", "MSFT", ...],
            "returns_data": {...}   # rendements historiques par actif
            "objective": "max_sharpe" | "min_volatility"  (optionnel)
        }

    Retourne : {"AAPL": 0.4, "MSFT": 0.6, ...}

    TODO:
        - construire la matrice de covariance à partir de returns_data
        - utiliser scipy.optimize.minimize (SLSQP) avec contraintes
          somme(poids) = 1, poids >= 0
        - objectif : maximiser (rendement - taux_sans_risque) / volatilité
    """
    tickers = payload.get('tickers', [])
    if not tickers:
        raise ValueError("Aucun ticker fourni pour l'optimisation Markowitz")

    # Placeholder : répartition égale en attendant l'implémentation réelle
    n = len(tickers)
    return {ticker: round(1 / n, 4) for ticker in tickers}
