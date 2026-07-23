"""
Moteur de backtest à fenêtres glissantes (rolling window backtest).

Principe : au lieu d'évaluer un portefeuille sur les MÊMES données qui ont
servi à le construire (biais "in-sample", ce qui avantage artificiellement
Markowitz puisque c'est exactement ce qu'il maximise), on simule ce qui se
serait réellement passé si on avait suivi la méthode année après année,
sans jamais connaître le futur au moment de la décision.

Exemple avec des données 2020-2025 et min_train_years=2 :
    Fenêtre 1 : entraînement sur 2020-2021 -> test sur 2022
    Fenêtre 2 : entraînement sur 2020-2022 -> test sur 2023
    Fenêtre 3 : entraînement sur 2020-2023 -> test sur 2024
    (fenêtre "expanding" : l'entraînement s'allonge à chaque fois, comme
    un investisseur qui accumule de l'historique au fil du temps)

Conçu pour être GÉNÉRIQUE : fonctionne avec n'importe quelle fonction
d'optimisation (Markowitz, Black-Litterman, et plus tard le module ML),
du moment qu'elle respecte la signature :
    optimize_fn(train_returns, risk_free_rate=..., **kwargs) -> dict avec 'weights'
"""

import numpy as np
import pandas as pd

from app.services.metrics import compute_max_drawdown, compute_sortino_ratio

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# Découpage en fenêtres glissantes
# ============================================================

def get_rolling_windows(returns: pd.DataFrame, min_train_years: int = 2) -> list:
    """
    Découpe l'historique en fenêtres (entraînement, test) par année
    calendaire, avec un entraînement qui s'allonge à chaque fenêtre
    (expanding window).

    Args:
        returns: DataFrame des rendements journaliers, indexé par date
        min_train_years: nombre minimum d'années d'entraînement avant le
                         premier test (2 par défaut : il faut un minimum
                         d'historique pour que l'optimisation ait un sens)

    Returns:
        Liste de dicts : [{'train_returns', 'test_returns', 'train_years',
        'test_year'}, ...], un élément par fenêtre de test.
    """
    years = sorted(returns.index.year.unique())

    windows = []
    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]

        train_mask = returns.index.year.isin(train_years)
        test_mask = returns.index.year == test_year

        windows.append({
            "train_returns": returns[train_mask],
            "test_returns": returns[test_mask],
            "train_years": train_years,
            "test_year": test_year,
        })

    return windows


# ============================================================
# Performance RÉELLEMENT obtenue sur la période de test
# ============================================================

def compute_realized_performance(
    weights: dict, test_returns: pd.DataFrame, risk_free_rate: float
) -> dict:
    """
    Calcule la performance RÉELLEMENT obtenue en suivant des poids fixes
    sur la période de test (jamais vue au moment de l'optimisation).

    Contrairement aux métriques "ex-ante" (expected_return, volatility)
    renvoyées par optimize_markowitz()/optimize_black_litterman(), qui
    sont des ESTIMATIONS basées sur l'historique d'entraînement, ici on
    mesure ce qui s'est VRAIMENT passé.
    """
    tickers = list(test_returns.columns)
    w = np.array([weights.get(ticker, 0.0) for ticker in tickers])

    daily_portfolio_returns = test_returns[tickers].values @ w
    n_days = len(daily_portfolio_returns)

    if n_days == 0:
        return {
            "realized_return": 0.0,
            "realized_volatility": 0.0,
            "realized_sharpe": 0.0,
            "realized_sortino": 0.0,
            "max_drawdown": 0.0,
            "n_test_days": 0,
        }

    # Rendement annualisé par capitalisation (cohérent avec le reste du projet)
    cumulative_return = np.prod(1 + daily_portfolio_returns) - 1
    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

    annualized_volatility = daily_portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    sharpe = (
        (annualized_return - risk_free_rate) / annualized_volatility
        if annualized_volatility > 0
        else 0.0
    )

    sortino = compute_sortino_ratio(daily_portfolio_returns, annualized_return, risk_free_rate)
    max_dd = compute_max_drawdown(daily_portfolio_returns)

    return {
        "realized_return": round(float(annualized_return), 4),
        "realized_volatility": round(float(annualized_volatility), 4),
        "realized_sharpe": round(float(sharpe), 4),
        "realized_sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "n_test_days": n_days,
    }


# ============================================================
# Backtest complet pour UNE méthode
# ============================================================

def run_backtest(
    returns: pd.DataFrame,
    method_name: str,
    optimize_fn,
    risk_free_rate: float = 0.02,
    min_train_years: int = 2,
    optimize_kwargs_fn=None,
) -> dict:
    """
    Exécute un backtest à fenêtres glissantes pour une méthode
    d'optimisation donnée.

    Args:
        returns: DataFrame complet des rendements journaliers
        method_name: nom affiché de la méthode (ex. "Markowitz")
        optimize_fn: fonction d'optimisation à tester, doit accepter
                     (train_returns, risk_free_rate=..., **kwargs) et
                     retourner un dict contenant au moins 'weights'.
                     Compatible avec optimize_markowitz() et
                     optimize_black_litterman() sans modification.
        risk_free_rate: taux sans risque annuel
        min_train_years: nombre minimum d'années d'entraînement
        optimize_kwargs_fn: fonction optionnelle callable(window) -> dict,
                            pour fournir des arguments supplémentaires
                            spécifiques à chaque fenêtre (ex. end_prices
                            pour Black-Litterman, qui change à chaque
                            fenêtre puisque la fin de la période
                            d'entraînement change)

    Returns:
        dict avec :
            - 'summary': moyennes/écarts-types agrégés sur toutes les fenêtres
            - 'windows': détail fenêtre par fenêtre (poids, performance
              prédite vs réalisée)
    """
    windows = get_rolling_windows(returns, min_train_years=min_train_years)

    if not windows:
        n_years_available = len(returns.index.year.unique())
        raise ValueError(
            f"Pas assez de données pour un backtest : {n_years_available} "
            f"année(s) disponible(s), il en faut au moins "
            f"{min_train_years + 1} (min_train_years={min_train_years} + "
            f"1 année de test)."
        )

    results = []
    for window in windows:
        extra_kwargs = optimize_kwargs_fn(window) if optimize_kwargs_fn else {}

        train_result = optimize_fn(
            window["train_returns"], risk_free_rate=risk_free_rate, **extra_kwargs
        )
        weights = train_result["weights"]

        perf = compute_realized_performance(
            weights, window["test_returns"], risk_free_rate
        )

        results.append({
            "test_year": int(window["test_year"]),
            "train_years": [int(y) for y in window["train_years"]],
            "weights": weights,
            "predicted_return": train_result.get("expected_return"),
            "predicted_volatility": train_result.get("volatility"),
            "predicted_sharpe": train_result.get("sharpe_ratio"),
            **perf,
        })

    realized_sharpes = [r["realized_sharpe"] for r in results]
    realized_returns = [r["realized_return"] for r in results]
    realized_vols = [r["realized_volatility"] for r in results]
    realized_sortinos = [r["realized_sortino"] for r in results]
    max_drawdowns = [r["max_drawdown"] for r in results]

    win_rate = sum(1 for r in realized_returns if r > 0) / len(realized_returns)

    summary = {
        "method": method_name,
        "n_windows": len(results),
        "avg_realized_return": round(float(np.mean(realized_returns)), 4),
        "avg_realized_volatility": round(float(np.mean(realized_vols)), 4),
        "avg_realized_sharpe": round(float(np.mean(realized_sharpes)), 4),
        "std_realized_sharpe": round(float(np.std(realized_sharpes)), 4),
        "avg_realized_sortino": round(float(np.mean(realized_sortinos)), 4),
        "avg_max_drawdown": round(float(np.mean(max_drawdowns)), 4),
        "worst_max_drawdown": round(float(np.min(max_drawdowns)), 4),
        "win_rate": round(win_rate, 4),
    }

    return {"summary": summary, "windows": results}