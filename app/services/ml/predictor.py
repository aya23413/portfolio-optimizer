"""
Étapes 10-11 du pipeline ML : orchestration complète + prédiction finale.

C'est le SEUL point d'entrée que ml_black_litterman.py doit appeler
(run_ml_pipeline) — tout le détail (data_loader, feature_engineering,
preprocessing, feature_selection, trainer, evaluator, model_selector)
reste interne à ce package.

Logique de sélection de modèle : un modèle par ticker est entraîné (3
candidats chacun : Ridge/RF/GB), puis UNE SEULE famille de modèle est
retenue pour l'ensemble du portefeuille (moyenne du score composite sur
tous les tickers) — cohérent avec l'esprit "un modèle qui se comporte
bien globalement", pas un modèle différent et donc moins comparable par
actif. La prédiction finale par ticker réutilise l'instance déjà
entraînée (et déjà réglée par hyperparameter_tuning) de cette famille
pour CET actif précis.
"""

import warnings

import numpy as np
import pandas as pd

from app.services.ml.data_loader import download_ohlcv, get_ticker_frame
from app.services.ml.feature_engineering import compute_features, compute_latest_features
from app.services.ml.preprocessing import prepare_dataset, apply_scaler
from app.services.ml.trainer import train_all_models
from app.services.ml.evaluator import evaluate_all_models
from app.services.ml.model_selector import select_best_model
from app.services.ml.utils import MIN_HISTORY_DAYS, DEFAULT_PREDICTION_HORIZON_DAYS


def _train_ticker(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float,
    prediction_horizon_days: int,
    random_state: int,
    fast_mode: bool = False,
) -> dict:
    """
    Exécute la chaîne complète feature_engineering -> preprocessing ->
    trainer -> evaluator pour UN actif. Retourne None si l'historique est
    insuffisant (plutôt que de lever une exception qui interromprait tout
    le portefeuille pour un seul ticker problématique).
    """
    ohlcv_ticker = get_ticker_frame(ohlcv, ticker)
    if len(ohlcv_ticker) < MIN_HISTORY_DAYS:
        return None

    features = compute_features(
        ohlcv_ticker, market_returns, risk_free_rate, prediction_horizon_days
    )
    if len(features) < 60:  # trop peu d'observations pour un split train/test sensé
        return None

    X, y = prepare_dataset(features)
    trained = train_all_models(X, y, random_state=random_state, fast_mode=fast_mode)
    evaluated = evaluate_all_models(trained)

    latest_features = compute_latest_features(ohlcv_ticker, market_returns, risk_free_rate)

    return {"trained": trained, "evaluated": evaluated, "latest_features": latest_features}


def run_ml_pipeline(
    returns: pd.DataFrame,
    tickers: list = None,
    risk_free_rate: float = 0.02,
    prediction_horizon_days: int = DEFAULT_PREDICTION_HORIZON_DAYS,
    random_state: int = 42,
    fast_mode: bool = False,
) -> dict:
    """
    Exécute le pipeline ML complet sur un portefeuille de tickers.

    Args:
        returns: DataFrame des rendements journaliers déjà téléchargés
                 par l'app (data_collector.py) — sert uniquement à définir
                 la plage de dates et le proxy de marché interne (moyenne
                 des rendements du portefeuille, cohérent avec la même
                 limite déjà documentée pour black_litterman.py)
        tickers: liste de tickers (défaut : toutes les colonnes de `returns`)
        risk_free_rate: taux sans risque annuel
        prediction_horizon_days: horizon de prédiction (jours de bourse)
        random_state: graine aléatoire pour la reproductibilité
        fast_mode: si True (voir trainer.py::FAST_MODE_MODELS), entraîne
                   seulement Ridge + Gradient Boosting avec un tuning
                   minimal, au lieu des 6 modèles complets. Destiné aux
                   appels répétés (backtest.py, qui appelle ce pipeline
                   une fois PAR FENÊTRE glissante -> le mode complet
                   multiplierait le temps de calcul par le nombre de
                   fenêtres, incompatible avec un usage interactif).

    Returns:
        dict {
            'predicted_returns': pd.Series {ticker: rendement annualisé prédit},
            'model_selected': nom de la famille de modèle retenue globalement,
            'model_diagnostics': {ticker: {mae, rmse, r2, direction_accuracy}},
            'global_r2': R² moyen (utilisé pour la confiance dans ml_black_litterman.py),
            'skipped_tickers': tickers sans historique suffisant (aucune vue générée),
        }
        Retourne un résultat vide (predicted_returns=Series vide) si AUCUN
        ticker n'a assez d'historique -> ml_black_litterman.py se rabat
        alors sur l'équilibre de marché pur, sans lever d'exception.
    """
    tickers = tickers or list(returns.columns)
    start = returns.index.min().strftime("%Y-%m-%d")
    end = returns.index.max().strftime("%Y-%m-%d")

    ohlcv = download_ohlcv(tickers, start=start, end=end)

    # Proxy de marché interne (limite documentée, cohérente avec le reste
    # du projet -- voir black_litterman.py)
    market_returns = returns.mean(axis=1)

    per_ticker = {}
    skipped_tickers = []
    for ticker in tickers:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _train_ticker(
                ticker, ohlcv, market_returns, risk_free_rate,
                prediction_horizon_days, random_state, fast_mode=fast_mode,
            )
        if result is None:
            skipped_tickers.append(ticker)
        else:
            per_ticker[ticker] = result

    if not per_ticker:
        return {
            "predicted_returns": pd.Series(dtype=float),
            "model_selected": None,
            "model_diagnostics": {},
            "global_r2": None,
            "skipped_tickers": skipped_tickers,
        }

    # Sélection de la famille de modèle globalement la plus performante
    # (moyenne du score composite sur tous les tickers entraînés avec succès)
    composite_scores_by_model = {}
    for ticker, res in per_ticker.items():
        _, details = select_best_model(res["evaluated"], res["trained"]["y_test"])
        for model_name, d in details.items():
            composite_scores_by_model.setdefault(model_name, []).append(d["composite_score"])

    avg_scores = {name: float(np.mean(scores)) for name, scores in composite_scores_by_model.items()}
    global_best_model = max(avg_scores, key=avg_scores.get)

    # Prédiction finale par ticker, avec l'instance déjà entraînée du
    # modèle globalement sélectionné pour CET actif
    predicted_returns = {}
    model_diagnostics = {}
    r2_values = []

    for ticker, res in per_ticker.items():
        trained = res["trained"]
        evaluated = res["evaluated"][global_best_model]
        estimator = trained["models"][global_best_model]["estimator"]
        selected_features = trained["selected_features"]
        scaler = trained["scaler"]

        latest_row = res["latest_features"][selected_features].to_frame().T
        latest_scaled = apply_scaler(scaler, latest_row)
        predicted_horizon_return = float(estimator.predict(latest_scaled.values)[0])

        # Annualisation cohérente avec le reste du projet (capitalisation)
        periods_per_year = 252 / prediction_horizon_days
        annualized_return = (1.0 + predicted_horizon_return) ** periods_per_year - 1.0

        predicted_returns[ticker] = annualized_return
        model_diagnostics[ticker] = {
            "mae": round(float(evaluated["mae"]), 4),
            "rmse": round(float(evaluated["rmse"]), 4),
            "r2": round(float(evaluated["r2"]), 4),
            "direction_accuracy": round(float(evaluated["direction_accuracy"]), 4),
        }
        r2_values.append(evaluated["r2"])

    return {
        "predicted_returns": pd.Series(predicted_returns),
        "model_selected": global_best_model,
        "model_diagnostics": model_diagnostics,
        "global_r2": float(np.mean(r2_values)) if r2_values else None,
        "skipped_tickers": skipped_tickers,
    }