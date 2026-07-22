"""
Service d'optimisation de portefeuille par Machine Learning prédictif.

Contrairement à HRP (ml_optimizer.py), qui structure le portefeuille sans
jamais estimer de rendement, cette approche utilise un modèle supervisé
(Random Forest, scikit-learn) pour PRÉDIRE le rendement futur de chaque
actif à partir de caractéristiques passées (momentum, volatilité récente,
Sharpe récent). Ces rendements prédits remplacent ensuite la simple
moyenne historique dans le même optimiseur moyenne-variance que
Markowitz (maximisation du ratio de Sharpe sous contraintes).

C'est donc un Markowitz "augmenté" : la seule différence avec
markowitz.py est la SOURCE des rendements espérés (prédits par un modèle
plutôt que moyenne historique brute) — tout le reste (covariance,
contraintes, optimiseur scipy) est strictement identique.

Construction du jeu d'entraînement — approche à FENÊTRE GLISSANTE
QUOTIDIENNE (et non plus un exemple par année) :
Plutôt que de créer un seul exemple d'entraînement par (actif, année),
ce qui ne donnait que 15-20 exemples au total (bien trop peu pour du
Machine Learning), on fait glisser une fenêtre jour par jour sur tout
l'historique disponible. Pour chaque jour t (avec assez d'historique
avant et assez de jours après) :
    - les FEATURES sont calculées à partir des rendements des ~252 jours
      PRÉCÉDANT t (passé uniquement, aucune fuite d'information)
    - la CIBLE est le rendement annualisé réalisé sur les ~21 jours
      SUIVANT t (environ 1 mois de bourse)
On répète cette opération pour chaque actif et pour chaque jour glissant
de l'historique (avec un pas de quelques jours pour limiter le nombre
d'exemples très proches les uns des autres). Avec 5 actifs sur 5 ans
(~1250 jours de bourse), cela génère plusieurs centaines à plusieurs
milliers d'exemples d'entraînement, contre 15-20 avec l'ancienne approche
annuelle.

LIMITE IMPORTANTE à documenter dans le rapport : ces exemples se
CHEVAUCHENT dans le temps (la fenêtre du jour t et celle du jour t+1
partagent presque les mêmes données). Ils ne sont donc PAS statistiquement
indépendants, contrairement à l'hypothèse habituelle du Machine Learning
classique. C'est une pratique courante et acceptée en finance quantitative
(faute de mieux, les données de marché sont intrinsèquement séquentielles),
mais elle signifie que le nombre brut d'exemples surestime la quantité
réelle d'information indépendante contenue dans le jeu d'entraînement —
un point à nuancer honnêtement plutôt qu'à passer sous silence.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

from app.services.markowitz import (
    compute_mean_returns,
    compute_covariance_matrix,
    portfolio_return,
    portfolio_volatility,
    portfolio_sharpe_ratio,
    negative_sharpe_ratio,
    build_constraints,
    build_bounds,
)

TRADING_DAYS_PER_YEAR = 252
FEATURE_LOOKBACK_DAYS = 252   # historique nécessaire pour calculer les features (~1 an)
FORWARD_HORIZON_DAYS = 21     # horizon de prédiction (~1 mois de bourse)
SLIDING_STEP_DAYS = 5         # pas entre deux fenêtres successives (~1 semaine)
MIN_HISTORY_DAYS = 60         # historique minimum absolu pour calculer des features


# ============================================================
# Ingénierie des caractéristiques (features)
# ============================================================

def extract_features(daily_returns: pd.Series) -> dict:
    """
    Calcule les caractéristiques (features) d'un actif à partir de son
    historique de rendements journaliers, à un instant donné.

    Toutes les fenêtres sont des fenêtres GLISSANTES vers le passé
    (aucune fuite d'information vers le futur) :
        - momentum sur 3, 6 et 12 mois (rendement cumulé récent)
        - volatilité annualisée sur 12 mois
        - ratio de Sharpe naïf sur 12 mois (sans taux sans risque, à
          titre de signal relatif)

    Returns:
        dict de features, ou None si l'historique est trop court pour
        être fiable (moins de MIN_HISTORY_DAYS jours).
    """
    if len(daily_returns) < MIN_HISTORY_DAYS:
        return None

    def cumulative_return(series: pd.Series) -> float:
        return float(np.prod(1 + series) - 1)

    window_1y = daily_returns.tail(TRADING_DAYS_PER_YEAR)
    window_6m = daily_returns.tail(TRADING_DAYS_PER_YEAR // 2)
    window_3m = daily_returns.tail(TRADING_DAYS_PER_YEAR // 4)

    vol_1y = float(window_1y.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    return_1y = cumulative_return(window_1y)

    return {
        "trailing_return_1y": return_1y,
        "trailing_return_6m": cumulative_return(window_6m),
        "trailing_return_3m": cumulative_return(window_3m),
        "trailing_volatility_1y": vol_1y,
        "trailing_sharpe_1y": return_1y / vol_1y if vol_1y > 0 else 0.0,
    }


def compute_horizon_return(forward_returns: pd.Series, annualize: bool = True) -> float:
    """
    Rendement réalisé sur une fenêtre future (la cible que le modèle doit
    apprendre à prédire), par capitalisation. Annualisé par défaut pour
    rester sur la même échelle que le reste du projet (comparable à
    mean_returns de markowitz.py), même si l'horizon réel est ~1 mois.
    """
    n_days = len(forward_returns)
    if n_days == 0:
        return 0.0
    cumulative = np.prod(1 + forward_returns) - 1
    if not annualize:
        return float(cumulative)
    return float((1 + cumulative) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)


# ============================================================
# Construction du jeu d'entraînement (fenêtre glissante quotidienne)
# ============================================================

def build_training_dataset(
    returns: pd.DataFrame,
    lookback_days: int = FEATURE_LOOKBACK_DAYS,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    step_days: int = SLIDING_STEP_DAYS,
) -> tuple:
    """
    Construit le jeu d'entraînement en faisant glisser une fenêtre
    JOUR PAR JOUR (par pas de step_days) sur tout l'historique
    disponible, pour chaque actif. Génère typiquement plusieurs centaines
    à plusieurs milliers d'exemples, contre 15-20 avec une approche par
    année civile.

    Pour chaque position t (espacée de step_days) :
        - features calculées sur les lookback_days jours PRÉCÉDANT t
        - cible = rendement annualisé réalisé sur les horizon_days jours
          SUIVANT t

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers)
        lookback_days: nombre de jours d'historique pour les features
        horizon_days: horizon de prédiction en jours de bourse
        step_days: pas entre deux fenêtres successives (pour limiter le
                   chevauchement excessif entre exemples consécutifs)

    Returns:
        (X, y, feature_names)
    """
    feature_names = [
        "trailing_return_1y", "trailing_return_6m", "trailing_return_3m",
        "trailing_volatility_1y", "trailing_sharpe_1y",
    ]

    X_rows = []
    y_values = []

    for ticker in returns.columns:
        series = returns[ticker]
        n_days = len(series)

        # t parcourt toutes les positions où on a assez d'historique AVANT
        # (>= lookback_days, ou au moins MIN_HISTORY_DAYS) et assez de
        # jours APRÈS (>= horizon_days) pour calculer la cible
        start_t = min(lookback_days, MIN_HISTORY_DAYS)
        end_t = n_days - horizon_days

        for t in range(start_t, end_t, step_days):
            history_window = series.iloc[max(0, t - lookback_days):t]
            forward_window = series.iloc[t:t + horizon_days]

            features = extract_features(history_window)
            if features is None:
                continue

            target = compute_horizon_return(forward_window, annualize=True)

            X_rows.append([features[name] for name in feature_names])
            y_values.append(target)

    return np.array(X_rows), np.array(y_values), feature_names


# ============================================================
# Entraînement et prédiction
# ============================================================

def train_return_predictor(
    X: np.ndarray, y: np.ndarray, n_estimators: int = 200, random_state: int = 42
) -> RandomForestRegressor:
    """
    Entraîne un Random Forest à prédire le rendement futur (~1 mois,
    annualisé) d'un actif à partir de ses caractéristiques passées.

    Profondeur et taille de feuille limitées pour contenir le
    sur-apprentissage, malgré le nombre d'exemples désormais bien plus
    élevé qu'avec l'ancienne approche annuelle (voir docstring du module
    sur le chevauchement des fenêtres).
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_leaf=10,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict_expected_returns(
    returns: pd.DataFrame, model: RandomForestRegressor, feature_names: list
) -> pd.Series:
    """
    Prédit le rendement annualisé attendu de chaque actif pour la
    PROCHAINE période, à partir de ses caractéristiques les plus
    récentes disponibles dans returns (donc "aujourd'hui", la toute fin
    de l'historique fourni).
    """
    predictions = {}
    fallback_mean_returns = None

    for ticker in returns.columns:
        features = extract_features(returns[ticker])

        if features is None:
            if fallback_mean_returns is None:
                fallback_mean_returns = compute_mean_returns(returns, annualize=True)
            predictions[ticker] = float(fallback_mean_returns[ticker])
            continue

        X_pred = np.array([[features[name] for name in feature_names]])
        predictions[ticker] = float(model.predict(X_pred)[0])

    return pd.Series(predictions)


# ============================================================
# Fonction principale
# ============================================================

def optimize_ml_predictive(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Calcule le portefeuille optimal en utilisant des rendements PRÉDITS
    par un Random Forest (entraîné sur des fenêtres glissantes
    quotidiennes), réinjectés dans le même optimiseur moyenne-variance
    que Markowitz (maximisation du ratio de Sharpe).

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers)
        risk_free_rate: taux sans risque annuel
        n_estimators: nombre d'arbres du Random Forest
        random_state: graine aléatoire, pour la reproductibilité

    Returns:
        dict avec la même structure que optimize_markowitz(), plus :
            - 'predicted_returns': rendements prédits par le modèle,
              par ticker
            - 'n_training_samples': taille du jeu d'entraînement utilisé
              (fenêtres glissantes, voir limite sur le chevauchement
              dans la docstring du module)
            - 'model_used': False si repli sur la moyenne historique
              (historique insuffisant pour entraîner un modèle fiable)
    """
    tickers = list(returns.columns)
    n_assets = len(tickers)

    if n_assets == 0:
        raise ValueError("Aucun actif fourni pour l'optimisation.")

    X, y, feature_names = build_training_dataset(returns)

    if len(X) < 30:
        # Pas assez d'exemples pour un entraînement fiable, même avec
        # l'approche à fenêtre glissante -> repli explicite sur la
        # moyenne historique (comportement Markowitz)
        predicted_returns = compute_mean_returns(returns, annualize=True)
        model_used = False
    else:
        model = train_return_predictor(
            X, y, n_estimators=n_estimators, random_state=random_state
        )
        predicted_returns = predict_expected_returns(returns, model, feature_names)
        predicted_returns = predicted_returns.reindex(tickers)
        model_used = True

    cov_matrix = compute_covariance_matrix(returns, annualize=True)

    initial_weights = np.array([1.0 / n_assets] * n_assets)
    constraints = build_constraints(n_assets)
    bounds = build_bounds(n_assets, max_weight=1.0)

    result = minimize(
        fun=negative_sharpe_ratio,
        x0=initial_weights,
        args=(predicted_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"L'optimisation ML a échoué : {result.message}")

    optimal_weights = np.where(np.abs(result.x) < 1e-6, 0.0, result.x)

    final_return = portfolio_return(optimal_weights, predicted_returns)
    final_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    final_sharpe = portfolio_sharpe_ratio(
        optimal_weights, predicted_returns, cov_matrix, risk_free_rate
    )

    return {
        "weights": {
            ticker: round(float(weight), 4)
            for ticker, weight in zip(tickers, optimal_weights)
        },
        "expected_return": round(final_return, 4),
        "volatility": round(final_volatility, 4),
        "sharpe_ratio": round(final_sharpe, 4),
        "predicted_returns": {
            t: round(float(v), 4) for t, v in predicted_returns.items()
        },
        "n_training_samples": len(X),
        "model_used": model_used,
    }