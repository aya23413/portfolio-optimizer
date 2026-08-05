"""
Étape 4 du pipeline ML : sélection de features.

Entrée : ~22 features (voir feature_engineering.py).
Sortie : sous-ensemble restreint (top-k), pour limiter le surapprentissage
vu la taille réduite du dataset (5 tickers, quelques années d'historique).

Combine 3 méthodes complémentaires plutôt qu'une seule (chacune a ses
angles morts) :
    - Mutual Information : capte les relations non linéaires
    - Permutation Importance (sur un modèle simple, Ridge) : mesure
      l'impact réel sur la performance de prédiction, pas juste la
      corrélation brute
    - Filtre de corrélation : élimine les features quasi redondantes
      entre elles (ex. price_vs_sma20 et price_vs_ema20, très corrélées)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    """Score d'information mutuelle par feature (plus haut = plus informatif)."""
    scores = mutual_info_regression(X.values, y.values, random_state=random_state)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def permutation_importance_ranking(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    """
    Importance par permutation sur un Ridge entraîné rapidement (modèle
    volontairement simple ici : le but n'est pas la précision finale,
    juste un ranking des features les plus utiles).
    """
    model = Ridge(alpha=1.0)
    model.fit(X.values, y.values)
    result = permutation_importance(
        model, X.values, y.values, n_repeats=10, random_state=random_state, scoring="r2"
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def correlation_filter(X: pd.DataFrame, threshold: float = 0.9) -> list:
    """
    Retire les features quasi redondantes : pour chaque paire fortement
    corrélée (|corr| > threshold), garde seulement la première rencontrée
    dans l'ordre des colonnes.

    Returns:
        Liste des colonnes à CONSERVER (les doublons ont été retirés).
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [col for col in X.columns if col not in to_drop]


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 12,
    correlation_threshold: float = 0.9,
    random_state: int = 42,
) -> list:
    """
    Combine les 3 méthodes ci-dessus pour produire la liste finale des
    features retenues.

    Méthode : filtre d'abord les redondances (corrélation), puis classe
    les features restantes par score combiné (moyenne des rangs
    mutual-info + permutation-importance, plutôt qu'une moyenne brute des
    scores, qui mélangerait des échelles non comparables).

    Returns:
        Liste des top_k noms de colonnes à utiliser pour l'entraînement.
    """
    non_redundant = correlation_filter(X, threshold=correlation_threshold)
    X_filtered = X[non_redundant]

    mi_rank = mutual_information_ranking(X_filtered, y, random_state).rank(ascending=False)
    perm_rank = permutation_importance_ranking(X_filtered, y, random_state).rank(ascending=False)

    combined_rank = (mi_rank + perm_rank) / 2.0
    selected = combined_rank.sort_values().head(top_k).index.tolist()
    return selected