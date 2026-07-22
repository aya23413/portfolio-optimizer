"""
Service d'optimisation de portefeuille par Hierarchical Risk Parity (HRP).

Référence : López de Prado, M. (2016), "Building Diversified Portfolios
that Outperform Out-of-Sample", Journal of Portfolio Management.

Contrairement à Markowitz et Black-Litterman, qui optimisent une fonction
objectif (maximiser le ratio de Sharpe) via scipy.optimize, HRP ne calcule
AUCUNE estimation de rendement futur et ne résout AUCUN problème
d'optimisation au sens classique. C'est une différence fondamentale,
souvent présentée dans la littérature comme la principale force de HRP :
elle ne peut donc pas "se tromper" sur des rendements attendus mal
estimés (la source principale d'instabilité de Markowitz observée dans
ce projet).

Principe général en 3 étapes :
    1. Clustering hiérarchique : regrouper les actifs selon leur
       similarité de comportement (corrélation), sans notion de rendement
    2. Quasi-diagonalisation : réordonner la matrice de covariance selon
       la structure du clustering (les actifs similaires se retrouvent
       proches les uns des autres dans la matrice)
    3. Allocation récursive par bissection : répartir le capital du haut
       vers le bas de l'arbre, en donnant plus de poids aux groupes
       d'actifs les moins risqués (parité de risque), jamais en fonction
       d'un rendement espéré

Implémentation "à la main" avec scipy/numpy (clustering hiérarchique de
scipy.cluster.hierarchy), sans bibliothèque spécialisée.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from app.services.markowitz import (
    compute_mean_returns,
    compute_covariance_matrix,
    portfolio_return,
    portfolio_volatility,
    portfolio_sharpe_ratio,
)

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# ÉTAPE 1 : Clustering hiérarchique
# ============================================================

def compute_distance_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit une matrice de corrélation en matrice de distance, utilisable
    par un algorithme de clustering.

        distance(i, j) = sqrt(0.5 * (1 - correlation(i, j)))

    Deux actifs très corrélés (corrélation proche de 1) ont une distance
    proche de 0 (ils sont "voisins" dans l'arbre). Deux actifs peu
    corrélés (voire négativement) ont une distance plus grande.
    """
    return np.sqrt(0.5 * (1 - corr_matrix))


def build_linkage_matrix(distance_matrix: pd.DataFrame) -> np.ndarray:
    """
    Construit l'arbre de clustering hiérarchique (linkage) à partir de la
    matrice de distance, avec la méthode du lien moyen ("average
    linkage"), un choix standard pour HRP.

    Le résultat est une "linkage matrix" au format scipy : chaque ligne
    représente une fusion de deux clusters, dans l'ordre où elles se
    produisent.
    """
    condensed_distance = squareform(distance_matrix.values, checks=False)
    return linkage(condensed_distance, method="average")


# ============================================================
# ÉTAPE 2 : Quasi-diagonalisation
# ============================================================

def get_quasi_diag_order(link: np.ndarray) -> list:
    """
    Extrait l'ordre des actifs tel que déterminé par l'arbre de
    clustering (parcours des feuilles de gauche à droite), pour que les
    actifs similaires se retrouvent côte à côte.

    Cette étape "quasi-diagonalise" la matrice de covariance : une fois
    réordonnée selon cette liste, les valeurs les plus fortes (actifs
    corrélés) se regroupent visuellement autour de la diagonale.
    """
    link = link.astype(int)
    num_items = link[-1, 3]  # nombre total d'actifs d'origine

    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items

        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


# ============================================================
# ÉTAPE 3 : Allocation récursive par bissection
# ============================================================

def compute_cluster_variance(cov_matrix: pd.DataFrame, cluster_items: list) -> float:
    """
    Variance d'un sous-groupe (cluster) d'actifs, en utilisant une
    pondération par l'inverse de la variance individuelle à l'intérieur
    du cluster (approche "naïve risk parity" locale, standard dans
    l'algorithme HRP original).
    """
    cov_slice = cov_matrix.loc[cluster_items, cluster_items]
    inv_var_weights = 1.0 / np.diag(cov_slice.values)
    inv_var_weights /= inv_var_weights.sum()
    cluster_variance = np.dot(inv_var_weights, np.dot(cov_slice.values, inv_var_weights))
    return cluster_variance


def recursive_bisection(cov_matrix: pd.DataFrame, sorted_tickers: list) -> pd.Series:
    """
    Répartit le capital de façon récursive, en descendant l'arbre de
    clustering : à chaque division d'un groupe d'actifs en deux
    sous-groupes, le poids est réparti INVERSEMENT PROPORTIONNELLEMENT au
    risque de chaque sous-groupe (le sous-groupe le plus risqué reçoit
    moins de capital).

    C'est le cœur de "risk parity" : contrairement à Markowitz, il n'y a
    ICI AUCUNE notion de rendement attendu — uniquement une logique de
    répartition du risque selon la structure de corrélation.
    """
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]

    while len(clusters) > 0:
        # Découpe chaque cluster de taille > 1 en deux moitiés
        clusters = [
            cluster[start:end]
            for cluster in clusters
            for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]

        # Répartit le poids entre les paires de sous-clusters
        for i in range(0, len(clusters), 2):
            cluster_left = clusters[i]
            cluster_right = clusters[i + 1]

            var_left = compute_cluster_variance(cov_matrix, cluster_left)
            var_right = compute_cluster_variance(cov_matrix, cluster_right)

            # Allocation inversement proportionnelle au risque : le
            # sous-groupe le moins risqué reçoit la plus grande part
            alpha = 1.0 - var_left / (var_left + var_right)

            weights[cluster_left] *= alpha
            weights[cluster_right] *= (1.0 - alpha)

    return weights


# ============================================================
# Fonction principale
# ============================================================

def optimize_hrp(returns: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """
    Calcule le portefeuille selon la méthode Hierarchical Risk Parity.

    Contrairement à optimize_markowitz() et optimize_black_litterman(),
    AUCUNE estimation de rendement futur n'intervient dans le calcul des
    poids eux-mêmes (étapes 1 à 3). Le rendement moyen historique n'est
    utilisé QU'APRÈS COUP, uniquement pour rapporter la performance
    attendue du portefeuille obtenu (à des fins de comparaison avec les
    deux autres méthodes) — jamais pour décider de la répartition.

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers)
        risk_free_rate: taux sans risque annuel, utilisé uniquement pour
                        le calcul du ratio de Sharpe rapporté en sortie

    Returns:
        dict avec la même structure que optimize_markowitz() :
            - 'weights': poids par ticker (dict ticker -> float)
            - 'expected_return', 'volatility', 'sharpe_ratio' : performance
              historique attendue du portefeuille obtenu (calculée après
              coup, à titre indicatif/comparatif)
    """
    tickers = list(returns.columns)
    n_assets = len(tickers)

    if n_assets == 0:
        raise ValueError("Aucun actif fourni pour l'optimisation.")

    if n_assets < 2:
        # Avec un seul actif, il n'y a rien à "hiérarchiser" : 100% dessus.
        return {
            "weights": {tickers[0]: 1.0},
            "expected_return": round(
                float(compute_mean_returns(returns, annualize=True).iloc[0]), 4
            ),
            "volatility": round(
                float(returns[tickers[0]].std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 4
            ),
            "sharpe_ratio": 0.0,
        }

    # Étape 1 : clustering hiérarchique, à partir de la matrice de corrélation
    corr_matrix = returns.corr()
    distance_matrix = compute_distance_matrix(corr_matrix)
    link = build_linkage_matrix(distance_matrix)

    # Étape 2 : ordre quasi-diagonal des actifs
    sorted_indices = get_quasi_diag_order(link)
    sorted_tickers = [tickers[i] for i in sorted_indices]

    # Étape 3 : allocation récursive par bissection, sur la covariance
    # ANNUALISÉE (cohérent avec markowitz.py et black_litterman.py)
    cov_matrix = compute_covariance_matrix(returns, annualize=True)
    hrp_weights = recursive_bisection(cov_matrix, sorted_tickers)

    # Remise dans l'ordre d'origine des tickers (plus pratique à l'usage)
    hrp_weights = hrp_weights.reindex(tickers)

    # Performance rapportée à titre indicatif (calculée après coup,
    # PAS utilisée pour déterminer les poids)
    mean_returns = compute_mean_returns(returns, annualize=True)
    weights_array = hrp_weights.values

    final_return = portfolio_return(weights_array, mean_returns)
    final_volatility = portfolio_volatility(weights_array, cov_matrix)
    final_sharpe = portfolio_sharpe_ratio(
        weights_array, mean_returns, cov_matrix, risk_free_rate
    )

    return {
        "weights": {
            ticker: round(float(weight), 4)
            for ticker, weight in hrp_weights.items()
        },
        "expected_return": round(final_return, 4),
        "volatility": round(final_volatility, 4),
        "sharpe_ratio": round(final_sharpe, 4),
    }