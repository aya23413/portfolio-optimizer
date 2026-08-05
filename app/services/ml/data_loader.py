"""
Étape 1 du pipeline ML : chargement des données OHLCV.

Rôle STRICTEMENT limité à la donnée brute : téléchargement, cache,
nettoyage des dates, fusion multi-tickers. Aucune feature, aucun ML ici
— voir feature_engineering.py pour la suite de la chaîne.
"""

import os
import hashlib
from datetime import date

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cache", "ohlcv")


def _cache_path(tickers: list, start: str, end: str) -> str:
    """Chemin de cache déterministe à partir des paramètres de la requête."""
    key = f"{'-'.join(sorted(tickers))}_{start}_{end}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{digest}.parquet")


def download_ohlcv(
    tickers: list,
    start: str,
    end: str = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Télécharge l'OHLCV (Open/High/Low/Close/Volume) pour une liste de
    tickers, avec cache local (parquet) pour éviter de re-télécharger à
    chaque exécution du pipeline.

    Args:
        tickers: liste de symboles boursiers
        start: date de début 'YYYY-MM-DD'
        end: date de fin 'YYYY-MM-DD' (défaut : aujourd'hui)
        use_cache: si True, réutilise un cache existant s'il correspond
                   exactement aux mêmes tickers/dates

    Returns:
        DataFrame avec colonnes MultiIndex (champ, ticker), ex.
        ('Close', 'AAPL'), index = dates de bourse (Timestamp), sans
        doublons ni trous non justifiés.
    """
    end = end or date.today().isoformat()
    cache_file = _cache_path(tickers, start, end)

    if use_cache and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # prix ajustés des dividendes/splits, cohérent avec data_collector.py
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise ValueError(
            f"Aucune donnée retournée par yfinance pour {tickers} entre {start} et {end}."
        )

    cleaned = _clean_ohlcv(raw)

    if use_cache:
        cleaned.to_parquet(cache_file)

    return cleaned


def _clean_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimal : index trié, dates dupliquées supprimées, et
    forward-fill limité (2 jours max) pour combler d'éventuels jours
    fériés locaux mal alignés entre marchés — pas plus, pour ne pas
    masquer de vraies données manquantes.
    """
    cleaned = raw.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
    cleaned = cleaned.ffill(limit=2)
    return cleaned


def get_ticker_frame(ohlcv: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extrait un DataFrame simple-index (Open/High/Low/Close/Volume) pour
    UN ticker à partir du DataFrame multi-tickers renvoyé par
    download_ohlcv(). Pratique pour feature_engineering.py, qui travaille
    actif par actif.
    """
    fields = ["Open", "High", "Low", "Close", "Volume"]
    frame = pd.DataFrame({field: ohlcv[field][ticker] for field in fields if field in ohlcv})
    return frame.dropna(how="all")