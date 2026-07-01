"""
Service de collecte et de prétraitement des données financières.

À terme, ce module doit :
- récupérer les prix historiques (ex. via yfinance ou une API type Alpha Vantage) ;
- nettoyer les données (valeurs manquantes, alignement des dates) ;
- calculer les rendements journaliers/logarithmiques ;
- mettre en cache les résultats dans data/raw et data/processed.
"""

import pandas as pd


def fetch_historical_data(tickers: list, start: str = None, end: str = None) -> dict:
    """
    Récupère les cours de clôture historiques pour une liste de tickers.

    TODO: brancher une source réelle (ex. yfinance.download(tickers, start, end)).
    Pour l'instant, cette fonction sert de point d'entrée à implémenter.
    """
    raise NotImplementedError(
        "Connecter une source de données réelle (yfinance, Alpha Vantage, etc.)"
    )


def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """Calcule les rendements à partir d'un DataFrame de prix (colonnes = tickers)."""
    if method == 'log':
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def clean_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données : suppression des NaN, alignement des dates."""
    return prices.dropna(how='any')
