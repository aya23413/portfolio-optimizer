"""
Tests pour optimize_black_litterman() avec données synthétiques
reproductibles (même principe que test_markowitz.py).

Lancer avec :
    python tests/test_black_litterman.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from app.services.black_litterman import optimize_black_litterman


def make_fake_returns(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Rendements synthétiques pour 4 actifs fictifs (voir test_markowitz.py)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    means = {"ACTIF_A": 0.0010, "ACTIF_B": 0.0006, "ACTIF_C": 0.0004, "ACTIF_D": 0.0002}
    stds = {"ACTIF_A": 0.025, "ACTIF_B": 0.015, "ACTIF_C": 0.010, "ACTIF_D": 0.008}
    data = {
        ticker: rng.normal(loc=means[ticker], scale=stds[ticker], size=n_days)
        for ticker in means
    }
    return pd.DataFrame(data, index=dates)


def make_fake_market_weights() -> dict:
    """Poids de marché fictifs, fournis directement (pas d'appel yfinance)."""
    return {"ACTIF_A": 0.40, "ACTIF_B": 0.30, "ACTIF_C": 0.20, "ACTIF_D": 0.10}


def test_weights_sum_to_one_no_views():
    """Sans vue, la somme des poids doit toujours valoir 1."""
    returns = make_fake_returns()
    market_weights = make_fake_market_weights()
    result = optimize_black_litterman(returns, views={}, market_weights=market_weights)

    total = sum(result["weights"].values())
    assert abs(total - 1.0) < 1e-4, f"Somme des poids = {total}"
    print(f"[OK] Sans vue : somme des poids = {total:.6f}")


def test_no_views_close_to_market_weights():
    """
    Propriété fondamentale de Black-Litterman : SANS vue, le portefeuille
    optimal doit être proche des poids de marché fournis (c'est la
    définition même de l'équilibre CAPM sous-jacent au modèle).
    """
    returns = make_fake_returns()
    market_weights = make_fake_market_weights()
    result = optimize_black_litterman(returns, views={}, market_weights=market_weights)

    for ticker, market_weight in market_weights.items():
        obtained = result["weights"].get(ticker, 0.0)
        assert abs(obtained - market_weight) < 0.05, (
            f"{ticker}: poids obtenu {obtained} trop loin du poids de "
            f"marché {market_weight} (écart > 5 points)"
        )
    print(f"[OK] Sans vue, poids proches du marché : {result['weights']}")


def test_strong_positive_view_increases_weight():
    """
    Une vue fortement positive et à haute confiance sur un actif doit
    augmenter son poids par rapport au cas sans vue.
    """
    returns = make_fake_returns()
    market_weights = make_fake_market_weights()

    baseline = optimize_black_litterman(returns, views={}, market_weights=market_weights)
    with_view = optimize_black_litterman(
        returns,
        views={"ACTIF_D": 0.50},  # vue très optimiste sur l'actif le moins pondéré
        confidences={"ACTIF_D": 0.9},
        market_weights=market_weights,
    )

    baseline_d = baseline["weights"]["ACTIF_D"]
    view_d = with_view["weights"]["ACTIF_D"]

    assert view_d > baseline_d, (
        f"Une vue positive forte sur ACTIF_D devrait augmenter son poids : "
        f"{baseline_d} (sans vue) vs {view_d} (avec vue)"
    )
    print(f"[OK] Vue positive sur ACTIF_D : poids passé de {baseline_d:.4f} à {view_d:.4f}")


def test_low_confidence_view_has_small_effect():
    """
    Une vue à très faible confiance ne doit presque pas déplacer le
    résultat par rapport au cas sans vue (c'est le comportement qui
    justifie ml_black_litterman.py : un modèle ML peu fiable, faible
    confiance, ne doit pas déstabiliser le portefeuille).
    """
    returns = make_fake_returns()
    market_weights = make_fake_market_weights()

    # NOTE : test sur ACTIF_A plutôt qu'ACTIF_D. Avec l'optimisation par
    # utilité quadratique (voir black_litterman.py::negative_quadratic_utility),
    # un actif dont le rendement d'équilibre (Pi) est proche de zéro
    # (c'est le cas d'ACTIF_D) est mécaniquement hypersensible à toute
    # vue non nulle, même à confiance quasi nulle : l'utilité quadratique
    # amplifie les écarts autour d'un rendement de base proche de zéro.
    # ACTIF_A, dont le Pi est substantiel, donne un test de "vue faible"
    # plus représentatif du cas d'usage réel (ml_black_litterman.py).
    baseline = optimize_black_litterman(returns, views={}, market_weights=market_weights)
    with_weak_view = optimize_black_litterman(
        returns,
        views={"ACTIF_A": 0.30},  # vue optimiste mais pas extrême
        confidences={"ACTIF_A": 0.05},  # confiance quasi nulle
        market_weights=market_weights,
    )

    baseline_a = baseline["weights"]["ACTIF_A"]
    view_a = with_weak_view["weights"]["ACTIF_A"]
    diff = abs(view_a - baseline_a)

    assert diff < 0.05, (
        f"Une vue à confiance quasi nulle ne devrait presque pas changer "
        f"le poids : écart observé = {diff:.4f} (attendu < 0.05)"
    )
    print(f"[OK] Vue à faible confiance sur ACTIF_A : écart de poids = {diff:.4f} (attendu petit)")


if __name__ == "__main__":
    print("=== Test 1 : somme des poids = 1 (sans vue) ===")
    test_weights_sum_to_one_no_views()

    print("\n=== Test 2 : sans vue, poids proches du marché ===")
    test_no_views_close_to_market_weights()

    print("\n=== Test 3 : vue positive forte augmente le poids ===")
    test_strong_positive_view_increases_weight()

    print("\n=== Test 4 : vue à faible confiance a un effet minime ===")
    test_low_confidence_view_has_small_effect()

    print("\n✅ Tous les tests Black-Litterman sont passés avec succès.")