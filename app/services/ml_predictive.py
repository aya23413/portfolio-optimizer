"""
⚠️ MODULE NON UTILISÉ EN PRODUCTION — conservé à titre de trace du
cheminement méthodologique (voir rapport, chapitre 4).

Ce module a été la 1ère implémentation du volet Machine Learning de ce
projet : les rendements prédits par un modèle supervisé remplacent
directement la moyenne historique dans un optimiseur Markowitz classique
("ML + Markowitz").

Cette approche a été ABANDONNÉE au profit de ml_black_litterman.py
("ML + Black-Litterman", actuellement utilisé par l'application), pour
la raison suivante, démontrée empiriquement dans ce projet : le modèle
ML a un pouvoir prédictif quasi nul (R² ≈ 0.01-0.03), et Markowitz est
l'optimiseur le PLUS sensible aux erreurs d'estimation (voir la
concentration extrême à 100% sur un seul actif déjà observée avec les
rendements historiques bruts). Injecter un signal faible dans
l'optimiseur le plus fragile a produit des poids instables d'un run à
l'autre. Black-Litterman, via son mécanisme de confiance bayésien, gère
nativement ce cas de figure — d'où le changement d'architecture.

Ce fichier reste fonctionnel et testable indépendamment (voir
essai_markowitz.py pour le principe), mais aucune route Flask ne
l'appelle plus.

============================================================
Service d'optimisation de portefeuille par Machine Learning prédictif.

Contrairement à HRP (ml_optimizer.py), qui structure le portefeuille sans
jamais estimer de rendement, cette approche utilise un modèle supervisé
pour PRÉDIRE le rendement futur de chaque actif à partir de
caractéristiques passées. Ces rendements prédits remplacent ensuite la
simple moyenne historique dans le même optimiseur moyenne-variance que
Markowitz (maximisation du ratio de Sharpe sous contraintes).

C'est donc un Markowitz "augmenté" : la seule différence avec
markowitz.py est la SOURCE des rendements espérés (prédits par un modèle
plutôt que moyenne historique brute) — tout le reste (covariance,
contraintes, optimiseur scipy) est strictement identique.

============================================================
HISTORIQUE DES AMÉLIORATIONS DE CE MODULE (à documenter dans le rapport)
============================================================

v1 -> v2 : passage d'un jeu d'entraînement annuel (~20 exemples) à une
approche par fenêtre glissante quotidienne (~1000+ exemples).

v2 -> v3 : correction d'un bug d'annualisation appliquée deux fois,
produisant des prédictions extrêmes (>300%) — voir le calcul du
rendement cible, annualisé UNE SEULE FOIS, sur la prédiction finale.

v3 -> v4 (version actuelle) : 3 améliorations, dans cet ordre :
    1. Plus de features : RSI, croisement de moyennes mobiles, bêta par
       rapport à un proxy de marché (moyenne équipondérée des actifs du
       portefeuille — en l'absence d'indice de référence externe comme
       le S&P 500 dans ce projet, à documenter comme simplification)
    2. Gradient Boosting à la place de Random Forest
    3. Réglage des hyperparamètres par validation croisée TEMPORELLE
       (TimeSeriesSplit, PAS un K-Fold classique qui mélangerait passé
       et futur au sein des plis de validation — important à justifier
       dans le rapport : la validation croisée standard suppose des
       exemples indépendants, hypothèse violée par des données de
       marché séquentielles)

LIMITE IMPORTANTE, inchangée depuis v2, à documenter dans le rapport :
les exemples générés par fenêtre glissante se CHEVAUCHENT dans le temps
et ne sont donc pas statistiquement indépendants. Le nombre brut
d'exemples (~1000+) surestime la quantité réelle d'information
indépendante contenue dans le jeu d'entraînement.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
RSI_PERIOD = 14
MA_SHORT_WINDOW = 20
MA_LONG_WINDOW = 100


# ============================================================
# Proxy de marché (pour le calcul du bêta)
# ============================================================

def compute_market_proxy_returns(returns: pd.DataFrame) -> pd.Series:
    """
    Rendement journalier moyen ÉQUIPONDÉRÉ de tous les actifs du
    portefeuille, utilisé comme proxy de "marché" pour calculer un bêta.

    LIMITE à documenter dans le rapport : ce n'est pas un vrai indice de
    marché (type S&P 500), faute d'avoir intégré une source de données
    externe dans ce projet. C'est une approximation simple mais
    raisonnable — le bêta obtenu mesure la sensibilité d'un actif aux
    mouvements communs du portefeuille étudié, pas au marché global.
    """
    return returns.mean(axis=1)


# ============================================================
# Ingénierie des caractéristiques (features)
# ============================================================

def compute_rsi(daily_returns: pd.Series, period: int = RSI_PERIOD) -> float:
    """
    Relative Strength Index (RSI), indicateur technique classique mesurant
    la vitesse et l'ampleur des mouvements récents (sur-achat / survente).
    Calculé ici à partir des rendements journaliers (approximation
    standard quand on ne dispose pas des prix bruts séparés des hausses
    et des baisses).

        RSI = 100 - 100 / (1 + RS),  RS = gain moyen / perte moyenne

    Retourne 50 (valeur neutre) si l'historique est trop court.
    """
    if len(daily_returns) < period + 1:
        return 50.0

    recent = daily_returns.tail(period)
    gains = recent[recent > 0]
    losses = -recent[recent < 0]

    avg_gain = gains.mean() if len(gains) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_ma_crossover(
    daily_returns: pd.Series, short_window: int = MA_SHORT_WINDOW, long_window: int = MA_LONG_WINDOW
) -> float:
    """
    Signal de croisement de moyennes mobiles : écart relatif entre la
    moyenne mobile courte et la moyenne mobile longue d'un indice de prix
    reconstruit à partir des rendements.

        signal = MA_courte / MA_longue - 1

    Un signal positif indique une tendance récente à la hausse (la
    moyenne courte est au-dessus de la longue), un signal classique
    d'analyse technique. Retourne 0 (neutre) si l'historique est trop
    court.
    """
    if len(daily_returns) < long_window:
        return 0.0

    price_index = (1 + daily_returns).cumprod()
    ma_short = price_index.tail(short_window).mean()
    ma_long = price_index.tail(long_window).mean()

    if ma_long == 0:
        return 0.0

    return float(ma_short / ma_long - 1)


def compute_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Bêta de l'actif par rapport au proxy de marché (voir
    compute_market_proxy_returns) : sensibilité de l'actif aux
    mouvements communs du portefeuille.

        beta = cov(actif, marché) / variance(marché)

    beta > 1 : l'actif amplifie les mouvements du marché (plus volatil)
    beta < 1 : l'actif amortit les mouvements du marché (plus défensif)

    Retourne 1.0 (bêta neutre) si l'historique est trop court ou si le
    marché n'a montré aucune variance sur la période.
    """
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0

    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()

    if var == 0:
        return 1.0

    return float(cov / var)


def extract_features(daily_returns: pd.Series, market_returns: pd.Series = None) -> dict:
    """
    Calcule les caractéristiques (features) d'un actif à partir de son
    historique de rendements journaliers, à un instant donné.

    Toutes les fenêtres sont des fenêtres GLISSANTES vers le passé
    (aucune fuite d'information vers le futur) :
        - momentum sur 3, 6 et 12 mois (rendement cumulé récent)
        - volatilité annualisée sur 12 mois
        - ratio de Sharpe naïf sur 12 mois
        - RSI 14 jours (sur-achat / survente)
        - croisement de moyennes mobiles 20/100 jours (tendance)
        - bêta par rapport au proxy de marché (sensibilité relative)

    Args:
        daily_returns: historique de rendements journaliers de l'actif
        market_returns: historique de rendements du proxy de marché,
                        aligné sur le même index que daily_returns.
                        Si None, le bêta retourné vaut 1.0 (neutre).

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

    beta = (
        compute_beta(daily_returns, market_returns)
        if market_returns is not None
        else 1.0
    )

    return {
        "trailing_return_1y": return_1y,
        "trailing_return_6m": cumulative_return(window_6m),
        "trailing_return_3m": cumulative_return(window_3m),
        "trailing_volatility_1y": vol_1y,
        "trailing_sharpe_1y": return_1y / vol_1y if vol_1y > 0 else 0.0,
        "rsi_14": compute_rsi(daily_returns),
        "ma_crossover": compute_ma_crossover(daily_returns),
        "beta": beta,
    }


def compute_horizon_return(forward_returns: pd.Series, annualize: bool = True) -> float:
    """
    Rendement réalisé sur une fenêtre future (la cible que le modèle doit
    apprendre à prédire), par capitalisation.
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
    disponible, pour chaque actif.

    IMPORTANT : les exemples sont générés dans l'ordre CHRONOLOGIQUE
    (actif par actif, puis jour par jour croissant), pour permettre une
    validation croisée temporelle (TimeSeriesSplit) en aval, qui suppose
    un ordre séquentiel cohérent.

    Returns:
        (X, y, feature_names)
    """
    feature_names = [
        "trailing_return_1y", "trailing_return_6m", "trailing_return_3m",
        "trailing_volatility_1y", "trailing_sharpe_1y",
        "rsi_14", "ma_crossover", "beta",
    ]

    market_returns = compute_market_proxy_returns(returns)

    X_rows = []
    y_values = []

    for ticker in returns.columns:
        series = returns[ticker]
        n_days = len(series)

        start_t = min(lookback_days, MIN_HISTORY_DAYS)
        end_t = n_days - horizon_days

        for t in range(start_t, end_t, step_days):
            history_window = series.iloc[max(0, t - lookback_days):t]
            market_window = market_returns.iloc[max(0, t - lookback_days):t]
            forward_window = series.iloc[t:t + horizon_days]

            features = extract_features(history_window, market_window)
            if features is None:
                continue

            target = compute_horizon_return(forward_window, annualize=False)

            X_rows.append([features[name] for name in feature_names])
            y_values.append(target)

    return np.array(X_rows), np.array(y_values), feature_names


# ============================================================
# Entraînement multi-modèles + sélection automatique
# ============================================================

# Les 3 familles de modèles comparées, chacune avec sa propre grille
# d'hyperparamètres. Choix volontairement diversifié : une méthode
# linéaire régularisée (Ridge) sert de référence simple — si les
# méthodes plus complexes ne la battent pas, c'est un résultat à
# documenter, pas à cacher.
MODEL_CONFIGS = {
    "ridge": {
        "estimator": lambda random_state: Ridge(random_state=random_state),
        "param_grid": {"alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
    },
    "random_forest": {
        "estimator": lambda random_state: RandomForestRegressor(
            random_state=random_state, min_samples_leaf=10
        ),
        "param_grid": {
            "n_estimators": [50, 100, 150],
            "max_depth": [2, 3, 4],
        },
    },
    "gradient_boosting": {
        "estimator": lambda random_state: GradientBoostingRegressor(
            random_state=random_state, min_samples_leaf=10, subsample=0.8
        ),
        "param_grid": {
            "n_estimators": [50, 100, 150],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    },
}


def train_single_model(X: np.ndarray, y: np.ndarray, model_type: str, random_state: int = 42) -> dict:
    """
    Entraîne UNE famille de modèle avec réglage des hyperparamètres par
    validation croisée temporelle (TimeSeriesSplit — voir justification
    dans la docstring du module).

    Returns:
        dict avec 'model' (meilleur estimateur), 'params' (meilleurs
        hyperparamètres), 'cv_score' (MAE moyen en validation croisée,
        SUR LE JEU D'ENTRAÎNEMENT UNIQUEMENT — ne regarde jamais le
        hold-out de test, pour une sélection de modèle sans fuite)
    """
    config = MODEL_CONFIGS[model_type]
    base_model = config["estimator"](random_state)
    tscv = TimeSeriesSplit(n_splits=5)

    search = GridSearchCV(
        base_model, config["param_grid"], cv=tscv,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    search.fit(X, y)

    return {
        "model": search.best_estimator_,
        "params": search.best_params_,
        "cv_score": float(-search.best_score_),  # MAE positif (plus petit = meilleur)
    }


def train_return_predictor(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """
    Entraîne les 3 familles de modèles (Ridge, Random Forest, Gradient
    Boosting), et sélectionne automatiquement la MEILLEURE sur la base
    du score de validation croisée obtenu PENDANT L'ENTRAÎNEMENT
    (jamais sur le jeu de test hold-out, qui reste réservé à l'évaluation
    finale du modèle gagnant — pas de fuite de sélection de modèle).

    Returns:
        (best_model, best_model_type, best_params, comparison_results)
        où comparison_results est un dict {model_type: {params, cv_score}}
        pour les 3 modèles, à des fins de transparence/rapport.
    """
    comparison_results = {}

    for model_type in MODEL_CONFIGS:
        result = train_single_model(X, y, model_type, random_state=random_state)
        comparison_results[model_type] = {
            "params": result["params"],
            "cv_mae": round(result["cv_score"], 4),
        }

    best_model_type = min(
        comparison_results, key=lambda mt: comparison_results[mt]["cv_mae"]
    )
    # Ré-entraîne le modèle gagnant pour récupérer l'objet estimé complet
    # (GridSearchCV ne garde en mémoire que le meilleur de CHAQUE famille,
    # on refait un seul appel léger pour le vainqueur global)
    best_result = train_single_model(X, y, best_model_type, random_state=random_state)

    return best_result["model"], best_model_type, best_result["params"], comparison_results


def evaluate_model_fit(
    X: np.ndarray, y: np.ndarray, model_type: str, random_state: int = 42, test_size: float = 0.2
) -> dict:
    """
    Évalue la qualité de prédiction du modèle GAGNANT (celui sélectionné
    par train_return_predictor) sur un sous-ensemble mis de côté
    (hold-out), avec les métriques standard de régression.

    Le split est fait de façon CHRONOLOGIQUE (shuffle=False) : le jeu de
    test correspond aux exemples les plus récents, jamais vus par le
    modèle pendant l'entraînement — cohérent avec la nature séquentielle
    des données et avec l'esprit du backtest hors-échantillon du projet.

    Returns:
        dict avec 'mae', 'rmse', 'r2', 'n_test_samples'
    """
    if len(X) < 10:
        return {"mae": None, "rmse": None, "r2": None, "n_test_samples": 0}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    config = MODEL_CONFIGS[model_type]
    diagnostic_model = config["estimator"](random_state)
    diagnostic_model.fit(X_train, y_train)
    y_pred = diagnostic_model.predict(X_test)

    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "n_test_samples": len(X_test),
    }


def predict_expected_returns(
    returns: pd.DataFrame, model, feature_names: list
) -> pd.Series:
    """
    Prédit le rendement annualisé attendu de chaque actif pour la
    PROCHAINE période, à partir de ses caractéristiques les plus
    récentes disponibles dans returns.
    """
    predictions = {}
    fallback_mean_returns = None
    market_returns = compute_market_proxy_returns(returns)

    for ticker in returns.columns:
        features = extract_features(returns[ticker], market_returns)

        if features is None:
            if fallback_mean_returns is None:
                fallback_mean_returns = compute_mean_returns(returns, annualize=True)
            predictions[ticker] = float(fallback_mean_returns[ticker])
            continue

        X_pred = np.array([[features[name] for name in feature_names]])
        predicted_horizon_return = float(model.predict(X_pred)[0])

        # Annualisation UNE SEULE FOIS, sur la prédiction finale (voir
        # historique des corrections en tête de module)
        annualized_prediction = (1 + predicted_horizon_return) ** (
            TRADING_DAYS_PER_YEAR / FORWARD_HORIZON_DAYS
        ) - 1
        predictions[ticker] = annualized_prediction

    return pd.Series(predictions)


# ============================================================
# Fonction principale
# ============================================================

def optimize_ml_predictive(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    random_state: int = 42,
) -> dict:
    """
    Calcule le portefeuille optimal en utilisant des rendements PRÉDITS
    par un Gradient Boosting Regressor (hyperparamètres réglés par
    validation croisée temporelle), réinjectés dans le même optimiseur
    moyenne-variance que Markowitz.

    Returns:
        dict avec la même structure que optimize_markowitz(), plus :
            - 'predicted_returns': rendements prédits par le modèle
            - 'n_training_samples': taille du jeu d'entraînement
            - 'model_used': False si repli sur la moyenne historique
            - 'model_diagnostics': MAE / RMSE / R² sur hold-out
            - 'best_hyperparameters': meilleurs paramètres trouvés par
              la validation croisée (transparence sur le réglage)
    """
    tickers = list(returns.columns)
    n_assets = len(tickers)

    if n_assets == 0:
        raise ValueError("Aucun actif fourni pour l'optimisation.")

    X, y, feature_names = build_training_dataset(returns)

    if len(X) < 30:
        predicted_returns = compute_mean_returns(returns, annualize=True)
        model_used = False
        model_diagnostics = {"mae": None, "rmse": None, "r2": None, "n_test_samples": 0}
        best_params = None
    else:
        model, best_params = train_return_predictor(X, y, random_state=random_state)
        predicted_returns = predict_expected_returns(returns, model, feature_names)
        predicted_returns = predicted_returns.reindex(tickers)
        model_used = True
        model_diagnostics = evaluate_model_fit(X, y, random_state=random_state)

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
        "model_diagnostics": model_diagnostics,
        "best_hyperparameters": best_params,
    }