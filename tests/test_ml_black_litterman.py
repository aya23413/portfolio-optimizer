"""
Tests pour optimize_ml_black_litterman() et le mécanisme de confiance
auto-calibrée sur le R².

Lancer avec :
    python tests/test_ml_black_litterman.py

Note : ces tests entraînent réellement des modèles ML (Ridge/RF/GB via
GridSearchCV), donc plus lents que les autres tests du projet (quelques
secondes à quelques dizaines de secondes selon la machine).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from app.services.ml_black_litterman import (
    r2_to_confidence,
    build_ml_views,
    optimize_ml_black_litterman,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
)


def make_fake_returns(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Rendements synthétiques pour 4 actifs, sur une période assez longue
    (n_days=800) pour dépasser le seuil minimum d'exemples d'entraînement
    du pipeline ML (voir build_training_dataset dans ml_predictive.py).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    means = {"ACTIF_A": 0.0010, "ACTIF_B": 0.0006, "ACTIF_C": 0.0004, "ACTIF_D": 0.0002}
    stds = {"ACTIF_A": 0.025, "ACTIF_B": 0.015, "ACTIF_C": 0.010, "ACTIF_D": 0.008}
    data = {
        ticker: rng.normal(loc=means[ticker], scale=stds[ticker], size=n_days)
        for ticker in means
    }
    return pd.DataFrame(data, index=dates)


def make_fake_prices(returns: pd.DataFrame) -> pd.Series:
    """Reconstruit un prix de fin de période à partir des rendements (base 100)."""
    price_index = 100 * (1 + returns).cumprod()
    return price_index.iloc[-1]


# ============================================================
# Tests de r2_to_confidence (fonction pure, rapide)
# ============================================================

def test_r2_to_confidence_bounds():
    """La confiance doit toujours rester dans [MIN_CONFIDENCE, MAX_CONFIDENCE]."""
    assert r2_to_confidence(-5.0) == MIN_CONFIDENCE, "R² très négatif -> confiance minimale"
    assert r2_to_confidence(0.0) == MIN_CONFIDENCE, "R² nul -> confiance minimale (borne basse)"
    assert r2_to_confidence(2.0) == MAX_CONFIDENCE, "R² > 1 (cas limite) -> confiance maximale"
    print(f"[OK] Bornes respectées : min={MIN_CONFIDENCE}, max={MAX_CONFIDENCE}")


def test_r2_to_confidence_monotonic():
    """Un meilleur R² doit toujours donner une confiance supérieure ou égale."""
    c_low = r2_to_confidence(0.05)
    c_mid = r2_to_confidence(0.30)
    c_high = r2_to_confidence(0.70)
    assert c_low <= c_mid <= c_high, "La confiance doit croître avec le R²"
    print(f"[OK] Monotonie : R²=0.05 -> {c_low}, R²=0.30 -> {c_mid}, R²=0.70 -> {c_high}")


# ============================================================
# Tests de build_ml_views
# ============================================================

def test_negative_predictions_excluded_by_default():
    """Les actifs à rendement prédit négatif ne doivent pas recevoir de vue."""
    predicted = pd.Series({"ACTIF_A": 0.15, "ACTIF_B": -0.05, "ACTIF_C": 0.08})
    views, confidences = build_ml_views(predicted, confidence=0.5, exclude_negative=True)

    assert "ACTIF_B" not in views, "ACTIF_B a un rendement négatif, ne devrait pas avoir de vue"
    assert "ACTIF_A" in views and "ACTIF_C" in views
    print(f"[OK] Exclusion des rendements négatifs : vues générées pour {list(views.keys())}")


def test_uniform_confidence_applied():
    """Toutes les vues générées doivent porter la même confiance (celle dérivée du R²)."""
    predicted = pd.Series({"ACTIF_A": 0.15, "ACTIF_C": 0.08})
    views, confidences = build_ml_views(predicted, confidence=0.42, exclude_negative=True)

    assert all(c == 0.42 for c in confidences.values())
    print(f"[OK] Confiance uniforme appliquée : {confidences}")


# ============================================================
# Test d'intégration (plus lent : entraîne réellement des modèles)
# ============================================================

def test_full_pipeline_runs_without_error():
    """
    Vérifie que le pipeline complet (features -> sélection de modèle ->
    vues -> Black-Litterman) s'exécute sans erreur et retourne un
    portefeuille valide (somme des poids = 1).
    """
    returns = make_fake_returns()
    end_prices = make_fake_prices(returns)
    market_weights = {"ACTIF_A": 0.40, "ACTIF_B": 0.30, "ACTIF_C": 0.20, "ACTIF_D": 0.10}

    result = optimize_ml_black_litterman(
        returns, end_prices=end_prices, market_weights=market_weights
    )

    total_weight = sum(result["weights"].values())
    assert abs(total_weight - 1.0) < 1e-3, f"Somme des poids = {total_weight}"
    assert "ml_model_selected" in result
    assert "confidence_used" in result
    assert 0.0 <= result["confidence_used"] <= 1.0

    print(f"[OK] Pipeline complet exécuté sans erreur.")
    print(f"     Modèle sélectionné : {result['ml_model_selected']}")
    print(f"     Confiance utilisée : {result['confidence_used']}")
    print(f"     Poids finaux : {result['weights']}")


if __name__ == "__main__":
    print("=== Test 1 : bornes de r2_to_confidence ===")
    test_r2_to_confidence_bounds()

    print("\n=== Test 2 : monotonie de r2_to_confidence ===")
    test_r2_to_confidence_monotonic()

    print("\n=== Test 3 : exclusion des prédictions négatives ===")
    test_negative_predictions_excluded_by_default()

    print("\n=== Test 4 : confiance uniforme appliquée aux vues ===")
    test_uniform_confidence_applied()

    print("\n=== Test 5 : pipeline complet (plus lent) ===")
    test_full_pipeline_runs_without_error()

    print("\n✅ Tous les tests ML + Black-Litterman sont passés avec succès.")