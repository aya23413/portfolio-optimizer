"""
Étape 3 du pipeline ML : preprocessing.

Entrée : toutes les features (feature_engineering.compute_features).
Sortie : dataset propre (NaN traités, outliers bornés, standardisé).

IMPORTANT (fuite de données) : le scaler doit être ajusté (fit) UNIQUEMENT
sur les données d'entraînement, puis appliqué (transform) tel quel sur le
test — jamais l'inverse. C'est pour ça que fit_scaler() et apply_scaler()
sont deux fonctions séparées plutôt qu'un simple fit_transform() global.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_incomplete_rows(features: pd.DataFrame, target_col: str = "future_return") -> pd.DataFrame:
    """
    Supprime les lignes avec NaN. En théorie, compute_features() a déjà
    fait un dropna() final, mais cette étape reste utile si plusieurs
    tickers aux historiques légèrement différents sont concaténés en
    amont (bourses fermées différemment selon les marchés, IPO récente...).
    """
    return features.dropna(subset=[target_col]).dropna()


def clip_outliers(
    features: pd.DataFrame,
    exclude_cols: list = None,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorisation : borne chaque colonne numérique à ses percentiles
    1%/99%, plutôt que de supprimer les lignes extrêmes (qui, en finance,
    sont souvent les plus informatives — ex. krach, plutôt du signal que
    du bruit). Seule la colonne cible peut être exclue si besoin (pour ne
    pas fausser l'évaluation finale du modèle sur les rendements réels).
    """
    exclude_cols = exclude_cols or []
    clipped = features.copy()
    numeric_cols = [c for c in clipped.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    for col in numeric_cols:
        lower = clipped[col].quantile(lower_pct)
        upper = clipped[col].quantile(upper_pct)
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)

    return clipped


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Ajuste un StandardScaler UNIQUEMENT sur les features d'entraînement."""
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Applique un scaler déjà ajusté (jamais re-fit ici)."""
    scaled = scaler.transform(X.values)
    return pd.DataFrame(scaled, index=X.index, columns=X.columns)


def prepare_dataset(
    features: pd.DataFrame,
    target_col: str = "future_return",
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> tuple:
    """
    Pipeline complet de preprocessing, jusqu'à la séparation X/y (SANS
    scaling, qui doit être fait après le split train/test pour éviter la
    fuite — voir fit_scaler/apply_scaler, appelés depuis trainer.py une
    fois le split TimeSeriesSplit effectué).

    Returns:
        X: DataFrame des features (non standardisées)
        y: Series de la cible (rendement futur, non transformée : on
           veut prédire un rendement réel, pas une valeur standardisée)
    """
    clean = drop_incomplete_rows(features, target_col)
    clean = clip_outliers(clean, exclude_cols=[target_col], lower_pct=lower_pct, upper_pct=upper_pct)

    y = clean[target_col]
    X = clean.drop(columns=[target_col])
    return X, y