"""
Étape 2 du pipeline ML : ingénierie des features.

Entrée : OHLCV d'un seul actif (voir data_loader.get_ticker_frame).
Sortie : DataFrame de features + colonne cible 'future_return'.

Noyau resserré (~22 features) plutôt que 40-60 : avec seulement 5 tickers
et quelques années d'historique, un trop grand nombre de variables mène
mécaniquement au surapprentissage (peu d'observations réellement
indépendantes vu l'autocorrélation des séries financières). Le
sous-ensemble ci-dessous couvre les mêmes 7 familles demandées
(rendements, momentum, tendance, volatilité, technique, facteurs,
statistiques, volume) sans redondance excessive.
"""

import numpy as np
import pandas as pd

from app.services.ml.utils import safe_divide, DEFAULT_PREDICTION_HORIZON_DAYS

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# Blocs de features individuels (chacun réutilisable/testable seul)
# ============================================================

def _returns_block(close: pd.Series) -> pd.DataFrame:
    """Rendements sur plusieurs horizons courts."""
    return pd.DataFrame({
        "return_1d": close.pct_change(1),
        "return_5d": close.pct_change(5),
        "return_21d": close.pct_change(21),
    })


def _momentum_block(close: pd.Series) -> pd.DataFrame:
    """
    Momentum sur horizons longs (1/3/6/12 mois, en jours de bourse).
    Différent du bloc "rendements" : sert à capter la tendance de fond,
    pas la dynamique court terme.
    """
    return pd.DataFrame({
        "momentum_1m": close.pct_change(21),
        "momentum_3m": close.pct_change(63),
        "momentum_6m": close.pct_change(126),
        "momentum_12m": close.pct_change(252),
    })


def _trend_block(close: pd.Series) -> pd.DataFrame:
    """
    Moyennes mobiles exprimées en écart RELATIF au prix (pas en niveau
    brut) : un niveau de prix absolu n'a aucun pouvoir prédictif d'un
    actif à l'autre, l'écart relatif si (ex. 'prix 8% au-dessus de sa
    SMA50' est comparable entre AAPL et NVDA, contrairement à un prix
    en dollars).
    """
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    return pd.DataFrame({
        "price_vs_sma20": safe_divide((close - sma20).values, sma20.values),
        "price_vs_sma50": safe_divide((close - sma50).values, sma50.values),
        "sma20_vs_sma50": safe_divide((sma20 - sma50).values, sma50.values),
        "price_vs_ema20": safe_divide((close - ema20).values, ema20.values),
    }, index=close.index)


def _volatility_block(daily_returns: pd.Series) -> pd.DataFrame:
    """Volatilité glissante (simple et pondérée exponentiellement)."""
    rolling_vol = daily_returns.rolling(21).std()
    ewma_vol = daily_returns.ewm(span=21, adjust=False).std()
    return pd.DataFrame({
        "rolling_vol_21d": rolling_vol,
        "ewma_vol_21d": ewma_vol,
    })


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index), calcul manuel (pas de dépendance TA-Lib)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = safe_divide(avg_gain.values, avg_loss.values, fill_value=0.0)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=close.index)


def _macd(close: pd.Series) -> pd.DataFrame:
    """MACD, ligne de signal, et histogramme (12/26/9, standard)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    })


def _bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Bandes de Bollinger, exprimées comme la POSITION relative du prix
    dans la bande (0 = bande basse, 1 = bande haute) plutôt qu'en niveau
    brut — plus directement exploitable par un modèle.
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    position = safe_divide((close - lower).values, (upper - lower).values, fill_value=0.5)
    return pd.DataFrame({"bollinger_position": position}, index=close.index)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range, normalisé par le prix (ATR% plutôt qu'ATR brut)."""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return pd.Series(safe_divide(atr.values, close.values), index=close.index)


def _factors_block(
    daily_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_daily: float,
    window: int = 63,
) -> pd.DataFrame:
    """
    Beta/alpha glissants vis-à-vis d'un proxy de marché, et ratios de
    performance ajustée au risque (Sharpe, Sortino) glissants.

    NOTE (limite documentée, cohérente avec le reste du projet) : le
    proxy de marché utilisé est la moyenne des rendements du portefeuille
    d'actifs suivis, pas un indice externe (S&P 500 par ex.) — voir
    limites déjà mentionnées pour black_litterman.py.
    """
    aligned = pd.concat([daily_returns, market_returns], axis=1).dropna()
    aligned.columns = ["asset", "market"]

    def _rolling_beta(sub: pd.DataFrame) -> float:
        if sub["market"].var() == 0 or len(sub) < 2:
            return np.nan
        cov = np.cov(sub["asset"], sub["market"])[0, 1]
        return cov / sub["market"].var()

    beta = pd.Series(index=daily_returns.index, dtype=float)
    for i in range(window, len(aligned) + 1):
        sub = aligned.iloc[i - window:i]
        beta.loc[aligned.index[i - 1]] = _rolling_beta(sub)
    beta = beta.reindex(daily_returns.index)

    rolling_mean_return = daily_returns.rolling(window).mean()
    rolling_market_mean = market_returns.rolling(window).mean()
    alpha = rolling_mean_return - beta * rolling_market_mean

    rolling_std = daily_returns.rolling(window).std()
    sharpe = safe_divide(
        (rolling_mean_return - risk_free_daily).values, rolling_std.values
    )

    downside = daily_returns.clip(upper=0)
    downside_std = downside.rolling(window).std()
    sortino = safe_divide(
        (rolling_mean_return - risk_free_daily).values, downside_std.values
    )

    return pd.DataFrame({
        "beta": beta,
        "alpha": alpha,
        "sharpe_rolling": sharpe,
        "sortino_rolling": sortino,
    }, index=daily_returns.index)


def _stats_block(close: pd.Series, daily_returns: pd.Series, window: int = 63) -> pd.DataFrame:
    """Drawdown glissant, asymétrie (skewness) et aplatissement (kurtosis)."""
    rolling_max = close.rolling(window).max()
    drawdown = safe_divide((close - rolling_max).values, rolling_max.values)
    skewness = daily_returns.rolling(window).skew()
    kurtosis = daily_returns.rolling(window).kurt()
    return pd.DataFrame({
        "drawdown_63d": drawdown,
        "skewness_63d": skewness,
        "kurtosis_63d": kurtosis,
    }, index=close.index)


def _volume_block(volume: pd.Series, window: int = 21) -> pd.DataFrame:
    """Volume moyen (log, pour stabiliser l'échelle) et ratio au volume récent."""
    avg_volume = volume.rolling(window).mean()
    volume_ratio = safe_divide(volume.values, avg_volume.values, fill_value=1.0)
    return pd.DataFrame({
        "log_avg_volume": np.log1p(avg_volume),
        "volume_ratio": volume_ratio,
    }, index=volume.index)


# ============================================================
# Assemblage complet
# ============================================================

def _compute_feature_blocks(
    ohlcv_ticker: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Calcule uniquement les FEATURES (sans label), factorisé pour être
    réutilisé à la fois par compute_features() (entraînement, avec label)
    et compute_latest_features() (prédiction live, sans label puisque le
    futur n'est par définition pas encore connu).
    """
    close = ohlcv_ticker["Close"]
    high = ohlcv_ticker["High"]
    low = ohlcv_ticker["Low"]
    volume = ohlcv_ticker["Volume"]
    daily_returns = close.pct_change()
    risk_free_daily = risk_free_rate / TRADING_DAYS_PER_YEAR

    blocks = [
        _returns_block(close),
        _momentum_block(close),
        _trend_block(close),
        _volatility_block(daily_returns),
        pd.DataFrame({"rsi_14": _rsi(close)}),
        _macd(close),
        _bollinger_bands(close),
        pd.DataFrame({"atr_pct": _atr(high, low, close)}),
        _factors_block(daily_returns, market_returns, risk_free_daily),
        _stats_block(close, daily_returns),
        _volume_block(volume),
    ]
    return pd.concat(blocks, axis=1)


def compute_features(
    ohlcv_ticker: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.02,
    prediction_horizon_days: int = DEFAULT_PREDICTION_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Calcule toutes les features pour UN actif, plus le label cible
    'future_return' (rendement à `prediction_horizon_days` jours,
    DÉCALÉ dans le futur -> les dernières lignes du DataFrame sont
    supprimées car leur cible n'est pas encore connue, pour éviter toute
    fuite d'information).

    Args:
        ohlcv_ticker: DataFrame Open/High/Low/Close/Volume d'UN actif
                      (voir data_loader.get_ticker_frame)
        market_returns: rendements journaliers du proxy de marché
                        (moyenne des actifs du portefeuille, calculée en
                        amont sur l'ensemble des tickers)
        risk_free_rate: taux sans risque ANNUEL
        prediction_horizon_days: horizon du label à prédire (en jours de
                                  bourse)

    Returns:
        DataFrame indexé par date, features + colonne 'future_return',
        sans NaN (lignes incomplètes en tête/queue supprimées).
    """
    close = ohlcv_ticker["Close"]
    features = _compute_feature_blocks(ohlcv_ticker, market_returns, risk_free_rate)

    # Label : rendement futur à horizon fixe, DÉCALÉ (shift négatif) pour
    # que la ligne à la date t contienne le rendement observé entre t et
    # t + horizon -- jamais d'information du futur dans les features elles-mêmes.
    features["future_return"] = close.pct_change(prediction_horizon_days).shift(
        -prediction_horizon_days
    )

    return features.dropna()


def compute_latest_features(
    ohlcv_ticker: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> pd.Series:
    """
    Calcule les features à la date LA PLUS RÉCENTE disponible, SANS label
    (contrairement à compute_features(), qui tronque les dernières lignes
    faute de label connu -> inutilisable pour une prédiction "aujourd'hui,
    que va-t-il se passer dans le futur ?"). C'est cette fonction qui sert
    à la prédiction live (voir predictor.py), pas compute_features().
    """
    features = _compute_feature_blocks(ohlcv_ticker, market_returns, risk_free_rate)
    return features.dropna().iloc[-1]