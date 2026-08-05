"""
Constantes et fonctions utilitaires partagées par tous les modules du
pipeline ML. Évite la duplication (ex. TRADING_DAYS_PER_YEAR redéfini
dans chaque fichier).
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Horizon de prédiction par défaut : rendement à 21 jours de bourse
# (~1 mois calendaire). Choix cohérent avec les fenêtres de backtest
# (expanding window, réévaluation annuelle) déjà utilisées dans le projet.
DEFAULT_PREDICTION_HORIZON_DAYS = 21

# Fenêtre minimale d'historique nécessaire pour calculer les features les
# plus longues (momentum 12 mois) + l'horizon de prédiction, en dessous
# de laquelle l'entraînement n'a pas de sens statistique.
MIN_HISTORY_DAYS = 320


def safe_divide(numerator, denominator, fill_value: float = 0.0):
    """
    Division élément par élément protégée contre les divisions par 0
    (fréquent avec des séries financières : volume nul, volatilité nulle
    sur un actif illiquide, etc.). Retourne fill_value plutôt que inf/NaN.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, fill_value)
    return result


def annualize_return(mean_daily_return: float) -> float:
    """Annualisation par capitalisation, cohérente avec markowitz.py."""
    return (1.0 + mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1.0


def annualize_volatility(daily_std: float) -> float:
    """Annualisation linéaire (racine du temps), cohérente avec markowitz.py."""
    return daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Standardisation glissante, utile pour certaines features techniques."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return safe_divide((series - rolling_mean).values, rolling_std.values)