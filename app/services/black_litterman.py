"""
Optimisation de portefeuille selon le modèle de Black-Litterman.

Combine les rendements d'équilibre du marché avec les anticipations
(vues) de l'investisseur pour produire une allocation plus stable
que Markowitz seul.
"""


def optimize(payload: dict) -> dict:
    """
    Calcule les poids optimaux selon Black-Litterman.

    payload attendu :
        {
            "tickers": [...],
            "returns_data": {...},
            "market_caps": {...},   # pondérations d'équilibre du marché
            "views": {...}          # vues subjectives de l'investisseur (optionnel)
        }

    TODO:
        - calculer les rendements implicites d'équilibre (reverse optimization)
        - intégrer les vues (matrice P, Q) et leur niveau de confiance (Omega)
        - combiner via la formule de Black-Litterman pour obtenir
          les rendements ajustés, puis optimiser (cf. markowitz.py)
    """
    tickers = payload.get('tickers', [])
    if not tickers:
        raise ValueError("Aucun ticker fourni pour l'optimisation Black-Litterman")

    n = len(tickers)
    return {ticker: round(1 / n, 4) for ticker in tickers}
