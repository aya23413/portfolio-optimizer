"""
Service d'optimisation de portefeuille selon le modèle de Markowitz
(théorie moderne du portefeuille, 1952).

Implémentation "à la main" avec scipy.optimize, sans bibliothèque
spécialisée (PyPortfolioOpt), à but pédagogique : chaque étape du calcul
(rendement espéré, covariance, ratio de Sharpe, contraintes, optimisation)
est isolée dans sa propre fonction documentée.

Principe général :
    On cherche les poids w = (w1, ..., wn) qui MAXIMISENT le ratio de Sharpe
    du portefeuille, sous les contraintes :
        - somme des poids = 1        (tout le capital est investi)
        - 0 <= wi <= 1 pour tout i   (pas de vente à découvert)

    Comme scipy.optimize.minimize() ne fait que minimiser, on minimise
    l'opposé du ratio de Sharpe (= -Sharpe), ce qui revient à le maximiser.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.services.data_collector import annualize_return

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# ÉTAPE 1 & 2 : Rendement moyen (espéré) par actif
# ============================================================

def compute_mean_returns(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Calcule le rendement moyen (espérance) de chaque actif à partir des
    rendements journaliers historiques.

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers)
        annualize: si True, annualise le rendement moyen journalier par
                   capitalisation : (1 + r_journalier)^252 - 1.
                   On réutilise annualize_return() de data_collector.py
                   pour garantir la MÊME méthode d'annualisation partout
                   dans le projet (cohérence entre la page Accueil, qui
                   affiche déjà ces statistiques, et l'optimiseur).

    Returns:
        pd.Series indexée par ticker : rendement moyen (journalier ou annualisé)
    """
    mean_daily = returns.mean()
    if annualize:
        return mean_daily.apply(annualize_return)
    return mean_daily


# ============================================================
# ÉTAPE 3 : Matrice de covariance
# ============================================================

def compute_covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Calcule la matrice de covariance des rendements journaliers, qui mesure
    comment les actifs varient ensemble (diversification).

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers)
        annualize: si True, annualise la covariance (cov_annuelle = cov_journalière * 252)

    Returns:
        DataFrame carré (ticker x ticker) de covariance

    Note sur la cohérence des méthodes d'annualisation :
        Contrairement au rendement moyen (annualisé par capitalisation,
        voir compute_mean_returns), la covariance est annualisée par une
        simple multiplication linéaire par 252. Ce n'est PAS une
        incohérence : sous l'hypothèse standard de rendements journaliers
        i.i.d., la variance d'une somme de rendements croît linéairement
        avec le nombre de jours (Var_annuelle = 252 * Var_journalière),
        donc la volatilité annualisée est proportionnelle à sqrt(252).
        C'est exactement la même formule que annualize_volatility() dans
        data_collector.py.
    """
    cov_daily = returns.cov()
    if annualize:
        return cov_daily * TRADING_DAYS_PER_YEAR
    return cov_daily


# ============================================================
# Performance d'un portefeuille donné un jeu de poids
# ============================================================

def portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    """
    Rendement espéré du portefeuille : moyenne pondérée des rendements
    individuels.
        R_portefeuille = somme(wi * Ri)
    """
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Volatilité (écart-type) du portefeuille, qui dépend à la fois des
    volatilités individuelles ET des covariances entre actifs (c'est ce
    terme qui capture l'effet de diversification) :
        variance_portefeuille = w^T . Cov . w
        volatilite_portefeuille = sqrt(variance_portefeuille)
    """
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return float(np.sqrt(variance))


def portfolio_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """
    Ratio de Sharpe : rendement excédentaire par unité de risque pris.
        Sharpe = (R_portefeuille - taux_sans_risque) / volatilite_portefeuille

    Plus il est élevé, meilleur est le couple rendement/risque du portefeuille.
    """
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol


# ============================================================
# ÉTAPE 4 : Fonction objectif (à minimiser)
# ============================================================

def negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """
    Oppose du ratio de Sharpe. scipy.optimize.minimize() ne sait que
    minimiser une fonction ; pour MAXIMISER le ratio de Sharpe, on
    minimise donc son opposé.
    """
    return -portfolio_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)


# ============================================================
# ÉTAPE 5 : Contraintes et bornes
# ============================================================

def build_constraints(n_assets: int) -> tuple:
    """
    Contrainte d'égalité : la somme des poids doit être égale à 1
    (tout le capital du portefeuille est investi, ni plus ni moins).

    scipy attend un dict avec :
        'type': 'eq'  -> contrainte d'égalité (fun(weights) doit valoir 0)
        'fun': la fonction qui doit s'annuler à l'optimum
    """
    return (
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
    )


def build_bounds(n_assets: int, max_weight: float = 1.0) -> tuple:
    """
    Bornes individuelles : chaque poids doit être compris entre 0 et max_weight.
    wi >= 0          interdit la vente à découvert (short selling).
    wi <= max_weight interdit l'effet de levier (par défaut 1.0, donc pas
                      de limite au-delà de 100 %) ET, si max_weight < 1.0,
                      impose une contrainte de diversification : aucun
                      actif ne peut dépasser cette part du portefeuille.
    """
    return tuple((0.0, max_weight) for _ in range(n_assets))


# ============================================================
# ÉTAPE 6 & 7 : Optimisation et résultat final
# ============================================================

def optimize_markowitz(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
) -> dict:
    """
    Calcule le portefeuille optimal au sens de Markowitz : celui qui
    maximise le ratio de Sharpe, sous contrainte que la somme des poids
    vaut 1 et qu'aucun poids n'est négatif ou supérieur à max_weight.

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers).
                 C'est le DataFrame déjà calculé par compute_returns()
                 dans data_collector.py.
        risk_free_rate: taux sans risque annuel utilisé dans le calcul du
                        ratio de Sharpe (par défaut 2 %, à ajuster selon
                        le taux des bons du Trésor US 3 mois par exemple).
        max_weight: poids maximum autorisé pour un seul actif (entre 0 et 1).
                    Par défaut 1.0 : Markowitz "pur", sans contrainte de
                    diversification (un actif peut recevoir 100 % du
                    portefeuille si c'est optimal). Une valeur comme 0.3
                    force une meilleure diversification en interdisant à
                    un actif de dépasser 30 % du portefeuille.

    Returns:
        dict avec :
            - 'weights': poids optimaux par ticker (dict ticker -> float)
            - 'expected_return': rendement annualisé attendu du portefeuille
            - 'volatility': volatilité annualisée du portefeuille
            - 'sharpe_ratio': ratio de Sharpe du portefeuille optimal
    """
    tickers = list(returns.columns)
    n_assets = len(tickers)

    if n_assets == 0:
        raise ValueError("Aucun actif fourni pour l'optimisation.")

    if not (0.0 < max_weight <= 1.0):
        raise ValueError("max_weight doit être compris entre 0 (exclu) et 1 (inclus).")

    # Vérification de faisabilité : si max_weight est trop restrictif,
    # il devient mathématiquement impossible que la somme des poids
    # atteigne 1 (ex. 5 actifs avec max_weight=0.15 -> somme max = 0.75).
    if n_assets * max_weight < 1.0:
        raise ValueError(
            f"max_weight={max_weight} est trop restrictif pour {n_assets} actifs : "
            f"la somme des poids ne peut pas dépasser {n_assets * max_weight:.2f}, "
            f"or elle doit valoir 1.0. Augmentez max_weight ou ajoutez des actifs."
        )

    # Étapes 1-3 : rendement moyen et covariance, annualisés
    mean_returns = compute_mean_returns(returns, annualize=True)
    cov_matrix = compute_covariance_matrix(returns, annualize=True)

    # Point de départ de l'optimisation : répartition égale
    # (c'est justement l'ancien placeholder, utilisé ici comme simple
    # point de départ pour l'algorithme, pas comme résultat final)
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Étape 5 : contraintes et bornes
    constraints = build_constraints(n_assets)
    bounds = build_bounds(n_assets, max_weight=max_weight)

    # Étape 6 : optimisation via SLSQP (Sequential Least Squares Programming),
    # méthode adaptée aux problèmes avec contraintes d'égalité et bornes.
    result = minimize(
        fun=negative_sharpe_ratio,
        x0=initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"L'optimisation a échoué : {result.message}")

    optimal_weights = result.x

    # Nettoyage : les poids quasi nuls (bruit numérique, ex. 1e-17) sont
    # ramenés à 0 pour un affichage propre.
    optimal_weights = np.where(np.abs(optimal_weights) < 1e-6, 0.0, optimal_weights)

    # Étape 7 : construction du résultat final
    final_return = portfolio_return(optimal_weights, mean_returns)
    final_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    final_sharpe = portfolio_sharpe_ratio(
        optimal_weights, mean_returns, cov_matrix, risk_free_rate
    )

    return {
        "weights": {
            ticker: round(float(weight), 4)
            for ticker, weight in zip(tickers, optimal_weights)
        },
        "expected_return": round(final_return, 4),
        "volatility": round(final_volatility, 4),
        "sharpe_ratio": round(final_sharpe, 4),
    }