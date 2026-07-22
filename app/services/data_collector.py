"""
Service de collecte et de prétraitement des données financières.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

from flask import current_app


def fetch_historical_data(tickers: list, start: str = None, end: str = None) -> dict:
    if not tickers:
        raise ValueError("Aucun ticker fourni")

    if not end:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    if not start:
        start = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError("Aucune donnée retournée par Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw[['Close']]
        prices.columns = tickers

    prices = clean_data(prices)

    if prices.empty:
        raise ValueError("Données vides après nettoyage.")

    returns = compute_returns(prices, method='log')
    _save_to_disk(prices, returns, tickers)

    # AJOUT : données formatées pour les graphiques
    chart_data = prepare_chart_data(prices, returns)

    summary = {
        'tickers': list(prices.columns),
        'start_date': str(prices.index.min().date()),
        'end_date': str(prices.index.max().date()),
        'nb_observations': int(len(prices)),
        'preview': prices.tail(5).round(2).reset_index().assign(
            Date=lambda d: d['Date'].astype(str)
        ).to_dict(orient='records'),
        'stats': {
            ticker: {
                'rendement_moyen_annualise': round(annualize_return(returns[ticker].mean()), 4),
                'volatilite_annualisee': round(annualize_volatility(returns[ticker].std()), 4),
            }
            for ticker in prices.columns
        },
        'chart_data': chart_data,  # AJOUT
    }
    return summary


def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    if method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def clean_data(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.dropna(how='any')


def annualize_return(daily_return: float, periods: int = 252) -> float:
    return (1 + daily_return) ** periods - 1


def annualize_volatility(daily_volatility: float, periods: int = 252) -> float:
    return daily_volatility * np.sqrt(periods)


def _save_to_disk(prices: pd.DataFrame, returns: pd.DataFrame, tickers: list) -> None:
    try:
        raw_dir = current_app.config['DATA_RAW_DIR']
        processed_dir = current_app.config['DATA_PROCESSED_DIR']
    except RuntimeError:
        raw_dir = os.path.join('data', 'raw')
        processed_dir = os.path.join('data', 'processed')

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    suffix = "_".join(tickers)[:100]
    prices.to_csv(os.path.join(raw_dir, f'prices_{suffix}.csv'))
    returns.to_csv(os.path.join(processed_dir, f'returns_{suffix}.csv'))


def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr().round(4)


def prepare_chart_data(prices_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    corr_matrix = compute_correlation_matrix(returns_df)
    return {
        "tickers": list(prices_df.columns),
        "prices": {
            "dates": prices_df.index.strftime("%Y-%m-%d").tolist(),
            "series": {t: prices_df[t].round(2).tolist() for t in prices_df.columns},
        },
        "returns": {
            "dates": returns_df.index.strftime("%Y-%m-%d").tolist(),
            "series": {t: returns_df[t].round(6).tolist() for t in returns_df.columns},
        },
        "correlation": {
            "labels": corr_matrix.columns.tolist(),
            "matrix": corr_matrix.values.tolist(),
        },
    }