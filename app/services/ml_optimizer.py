"""
Optimisation de portefeuille assistée par Machine Learning.

Le rôle du ML ici peut être, selon le choix méthodologique retenu :
- prédire les rendements futurs (ex. LSTM, Random Forest) pour alimenter
  ensuite une optimisation Markowitz avec des rendements "prédits" ;
- apprendre directement une politique d'allocation (Reinforcement Learning) ;
- ou effectuer un clustering des actifs pour améliorer la diversification.

Ce module doit être précisé une fois la technique choisie.
"""


def optimize(payload: dict) -> dict:
    """
    Calcule les poids optimaux via une approche ML.

    payload attendu :
        {
            "tickers": [...],
            "returns_data": {...},
            "model_params": {...}  # hyperparamètres du modèle (optionnel)
        }

    TODO:
        - charger/entraîner le modèle ML choisi
        - générer des rendements prédits ou une allocation directe
        - retourner les poids normalisés (somme = 1)
    """
    tickers = payload.get('tickers', [])
    if not tickers:
        raise ValueError("Aucun ticker fourni pour l'optimisation ML")

    n = len(tickers)
    return {ticker: round(1 / n, 4) for ticker in tickers}
