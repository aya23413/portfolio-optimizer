"""
Service combinant Machine Learning et Black-Litterman : le modèle prédictif
(voir ml_predictive.py) génère des VUES automatiques, injectées dans le
mécanisme bayésien de Black-Litterman plutôt que directement dans un
optimiseur moyenne-variance classique (Markowitz).

JUSTIFICATION SCIENTIFIQUE (à citer dans le rapport) :
Cette architecture est directement inspirée de travaux publiés sur le
sujet, notamment Manish & Chahal (2026), "Bridging behavioral insights
and quantitative finance: AI-powered Black-Litterman framework with
technical and sentiment signals", Research in International Business
and Finance, qui utilise des prévisions de deep learning comme vecteur
de vues pour Black-Litterman plutôt que des vues subjectives d'expert.

POURQUOI CETTE ARCHITECTURE EST PLUS ROBUSTE QUE "ML + MARKOWITZ" :
Le module ml_predictive.py a démontré empiriquement un R² proche de 0
(pouvoir prédictif quasi nul du modèle). Injecter directement des
prédictions bruitées dans Markowitz (l'optimiseur le PLUS sensible aux
erreurs d'estimation, comme démontré par la concentration à 100% sur un
seul actif observée dans ce projet) amplifie ce bruit. Black-Litterman,
via sa matrice d'incertitude Omega, est conçu pour ABSORBER une vue peu
fiable en la ramenant vers l'équilibre de marché plutôt que de la
laisser dominer le portefeuille — l'outil mathématiquement approprié
pour un signal faible, pas un contournement.

MÉCANISME DE CONFIANCE AUTO-CALIBRÉE (contribution propre à ce projet) :
Plutôt que de fixer arbitrairement un niveau de confiance pour chaque
vue générée par le modèle, la confiance est dérivée directement du R²
mesuré empiriquement sur le jeu de test hold-out (voir
ml_predictive.evaluate_model_fit) :

    confiance = clip(R², confiance_min, confiance_max)

Un modèle qui prédit bien (R² élevé) génère des vues auxquelles
Black-Litterman fait davantage confiance ; un modèle qui ne prédit
quasiment rien (R² proche de 0, le cas mesuré dans ce projet) génère des
vues à très faible confiance, ramenant le résultat près du portefeuille
d'équilibre de marché pur — un comportement scientifiquement cohérent,
piloté par la donnée elle-même plutôt que par un réglage manuel.

RÈGLE D'ABSTENTION (inspirée de Manish & Chahal, 2026) :
Les actifs dont le rendement prédit est NÉGATIF ne reçoivent PAS de vue
explicite (on ne "parie" pas activement sur une baisse) — Black-Litterman
s'appuie alors uniquement sur l'équilibre de marché pour cet actif,
plutôt que sur une prédiction négative potentiellement peu fiable.
"""

import numpy as np
import pandas as pd

from app.services.ml_predictive import (
    build_training_dataset,
    train_return_predictor,
    evaluate_model_fit,
    predict_expected_returns,
)
from app.services.black_litterman import optimize_black_litterman

MIN_CONFIDENCE = 0.05  # confiance minimale, même si R² est négatif ou nul
MAX_CONFIDENCE = 0.95  # confiance maximale, jamais totale (incertitude résiduelle)


def r2_to_confidence(r2: float) -> float:
    """
    Convertit un R² (potentiellement négatif, un modèle peut faire pire
    que la moyenne) en niveau de confiance utilisable par Black-Litterman,
    borné dans [MIN_CONFIDENCE, MAX_CONFIDENCE].

    Un R² négatif ou nul -> confiance minimale (le modèle n'apporte
    aucune information exploitable, laisser parler l'équilibre de marché).
    Un R² proche de 1 -> confiance maximale (rare en pratique sur des
    rendements financiers, voir limites documentées dans ml_predictive.py).
    """
    return float(np.clip(r2, MIN_CONFIDENCE, MAX_CONFIDENCE))


def build_ml_views(
    predicted_returns: pd.Series, confidence: float, exclude_negative: bool = True
) -> tuple:
    """
    Convertit les rendements prédits par le modèle ML en vues Black-
    Litterman (Q) avec un niveau de confiance uniforme dérivé du R².

    Args:
        predicted_returns: rendements annualisés prédits par le modèle,
                            par ticker (sortie de ml_predictive.predict_expected_returns)
        confidence: niveau de confiance à appliquer à toutes les vues
                    (dérivé du R² via r2_to_confidence)
        exclude_negative: si True (comportement par défaut, inspiré de
                          Manish & Chahal 2026), les actifs à rendement
                          prédit négatif ne reçoivent PAS de vue —
                          Black-Litterman se rabat sur l'équilibre de
                          marché pour ces actifs plutôt que de "parier"
                          activement sur une baisse

    Returns:
        (views, confidences) : deux dicts compatibles avec
        black_litterman.optimize_black_litterman()
    """
    views = {}
    confidences = {}

    for ticker, predicted_return in predicted_returns.items():
        if exclude_negative and predicted_return < 0:
            continue
        views[ticker] = float(predicted_return)
        confidences[ticker] = confidence

    return views, confidences


def optimize_ml_black_litterman(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    tau: float = 0.05,
    end_prices: pd.Series = None,
    market_weights: dict = None,
    random_state: int = 42,
    exclude_negative_views: bool = True,
) -> dict:
    """
    Calcule le portefeuille optimal en combinant :
        1. Le pipeline prédictif de ml_predictive.py (sélection
           automatique parmi Ridge / Random Forest / Gradient Boosting)
        2. Le mécanisme bayésien de Black-Litterman, où les prédictions
           du modèle deviennent des vues, avec une confiance auto-
           calibrée sur le R² mesuré empiriquement

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers)
        risk_free_rate: taux sans risque annuel
        tau: paramètre d'échelle de l'incertitude sur Pi (voir black_litterman.py)
        end_prices: prix de fin de période historique, pour la cohérence
                    temporelle des poids de marché (voir black_litterman.py)
        market_weights: poids de marché fournis directement (sinon
                        calculés automatiquement via capitalisation)
        random_state: graine aléatoire pour la reproductibilité du modèle ML
        exclude_negative_views: voir build_ml_views()

    Returns:
        dict avec la même structure que optimize_black_litterman(), plus :
            - 'ml_predicted_returns': rendements prédits par le modèle
            - 'ml_model_selected': famille de modèle retenue (ridge/
              random_forest/gradient_boosting)
            - 'ml_model_diagnostics': MAE/RMSE/R² du modèle sur hold-out
            - 'confidence_used': niveau de confiance appliqué aux vues
              (dérivé du R²)
            - 'views_excluded': liste des tickers exclus des vues
              (rendement prédit négatif)
    """
    tickers = list(returns.columns)

    # Étape 1 : pipeline prédictif (identique à ml_predictive.py)
    X, y, feature_names = build_training_dataset(returns)

    if len(X) < 30:
        # Pas assez de données pour un modèle fiable -> aucune vue,
        # Black-Litterman se comporte comme un portefeuille d'équilibre pur
        predicted_returns = pd.Series(dtype=float)
        model_selected = None
        model_diagnostics = {"mae": None, "rmse": None, "r2": None, "n_test_samples": 0}
        confidence = MIN_CONFIDENCE
    else:
        model, model_selected, best_params, comparison = train_return_predictor(
            X, y, random_state=random_state
        )
        predicted_returns = predict_expected_returns(returns, model, feature_names)
        predicted_returns = predicted_returns.reindex(tickers)
        model_diagnostics = evaluate_model_fit(
            X, y, model_type=model_selected, random_state=random_state
        )
        r2 = model_diagnostics.get("r2")
        confidence = r2_to_confidence(r2 if r2 is not None else 0.0)

    # Étape 2 : conversion des prédictions en vues Black-Litterman
    if len(predicted_returns) > 0:
        views, confidences = build_ml_views(
            predicted_returns, confidence, exclude_negative=exclude_negative_views
        )
    else:
        views, confidences = {}, {}

    excluded_tickers = [
        t for t in predicted_returns.index if t not in views
    ] if len(predicted_returns) > 0 else []

    # Étape 3 : Black-Litterman avec ces vues auto-générées (réutilise
    # intégralement le moteur existant, aucune duplication de code)
    result = optimize_black_litterman(
        returns,
        views=views,
        confidences=confidences,
        risk_free_rate=risk_free_rate,
        tau=tau,
        market_weights=market_weights,
        end_prices=end_prices,
    )

    result["ml_predicted_returns"] = {
        t: round(float(v), 4) for t, v in predicted_returns.items()
    }
    result["ml_model_selected"] = model_selected
    result["ml_model_diagnostics"] = model_diagnostics
    result["confidence_used"] = round(confidence, 4)
    result["views_excluded"] = excluded_tickers

    return result