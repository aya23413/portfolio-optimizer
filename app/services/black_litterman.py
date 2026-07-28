"""
Service d'optimisation de portefeuille selon le modèle de Black-Litterman
(Black & Litterman, 1992).

Contrairement à Markowitz "pur" qui part uniquement des rendements
historiques moyens (souvent bruités, d'où la sur-concentration observée),
Black-Litterman part d'un point d'ancrage plus stable : les rendements
D'ÉQUILIBRE DU MARCHÉ (déduits des poids de capitalisation boursière),
puis les ajuste selon des VUES d'investisseur explicites, avec un niveau
de confiance associé à chaque vue.

Principe général en 4 étapes :
    1. Poids de marché (capitalisation boursière) -> point d'ancrage neutre
    2. Rendements d'équilibre implicites (Pi), par optimisation inverse
    3. Intégration des vues de l'investisseur (formule bayésienne de Black-Litterman)
    4. Optimisation moyenne-variance classique (max Sharpe) sur le résultat

Implémentation "à la main" avec numpy/scipy, sans bibliothèque spécialisée,
en réutilisant les briques déjà écrites dans markowitz.py (performance
de portefeuille, contraintes, optimiseur) pour éviter la duplication.
"""

import numpy as np
import pandas as pd
import yfinance as yf

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
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_AVERSION = 2.5  # valeur standard dans la littérature si le calcul échoue

# Entreprises à classes d'actions multiples (Classe A/B/C...), pour
# lesquelles yfinance ne renvoie souvent que le nombre d'actions d'UNE
# seule classe via 'sharesOutstanding' -> capitalisation boursière et
# poids de marché sous-estimés dans get_market_cap_weights().
# Confirmé empiriquement sur GOOGL (voir rapport, chapitre 4).
MULTI_CLASS_TICKERS = {"GOOGL", "GOOG", "META", "BRK.A", "BRK.B", "BRKA", "BRKB"}


# ============================================================
# ÉTAPE 1 : Poids de marché (capitalisation boursière)
# ============================================================

def get_market_cap_weights(tickers: list, end_prices: pd.Series = None) -> dict:
    """
    Calcule les poids d'équilibre du marché (proportionnels à la
    capitalisation boursière).

    IMPORTANT — cohérence temporelle avec Markowitz :
    yf.Ticker(ticker).info['marketCap'] renvoie la capitalisation
    D'AUJOURD'HUI, ce qui n'a pas de sens si vos rendements analysés
    couvrent une période historique différente (ex. 2020-2025) : Markowitz
    serait comparé à un Black-Litterman construit avec des informations
    "du futur" par rapport à la fenêtre étudiée.

    On calcule donc ici une capitalisation approximative À LA FIN DE LA
    PÉRIODE HISTORIQUE analysée :
        capitalisation ≈ prix_de_clôture_fin_de_période × actions_en_circulation

    Le nombre d'actions en circulation est récupéré via yfinance (valeur
    actuelle, une approximation raisonnable pour des méga-capitalisations
    sur quelques années, les rachats d'actions/émissions étant marginaux
    par rapport aux variations de prix — à documenter comme limite dans
    le rapport).

    Args:
        tickers: liste de symboles boursiers
        end_prices: pd.Series {ticker: prix de clôture à la fin de la
                    période historique analysée}. Si None, revient au
                    comportement précédent (capitalisation temps réel,
                    à éviter si on compare avec un Markowitz historique).

    Returns:
        dict {ticker: poids de marché}, somme des valeurs = 1.0
    """
    market_caps = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            shares_outstanding = info.get("sharesOutstanding")

            if shares_outstanding and end_prices is not None and ticker in end_prices:
                # Capitalisation historique approximée : prix de fin de
                # période x actions en circulation (actuelles)
                market_caps[ticker] = end_prices[ticker] * shares_outstanding
            else:
                # Repli : capitalisation temps réel (moins cohérent
                # temporellement, mais mieux que rien)
                cap = info.get("marketCap")
                if cap:
                    market_caps[ticker] = cap
        except Exception:
            pass

    if len(market_caps) != len(tickers):
        missing = set(tickers) - set(market_caps.keys())
        print(
            f"[Black-Litterman] Capitalisation boursière indisponible pour "
            f"{missing} -> repli sur une pondération égale pour tous les actifs."
        )
        n = len(tickers)
        return {ticker: 1.0 / n for ticker in tickers}

    total = sum(market_caps.values())
    return {ticker: cap / total for ticker, cap in market_caps.items()}


# ============================================================
# ÉTAPE 2 : Rendements d'équilibre implicites (reverse optimization)
# ============================================================

def estimate_risk_aversion(
    market_weights: dict,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """
    Coefficient d'aversion au risque du marché (delta), estimé à partir de
    la performance historique du portefeuille de marché pondéré par
    capitalisation :
        delta = (rendement_marché - taux_sans_risque) / variance_marché

    Plus delta est élevé, plus le marché "exige" un rendement élevé pour
    un niveau de risque donné (aversion au risque forte).
    """
    weights = np.array([market_weights[t] for t in mean_returns.index])
    market_return = portfolio_return(weights, mean_returns)
    market_variance = portfolio_volatility(weights, cov_matrix) ** 2

    if market_variance == 0:
        return DEFAULT_RISK_AVERSION
    return (market_return - risk_free_rate) / market_variance


def compute_implied_equilibrium_returns(
    market_weights: dict,
    cov_matrix: pd.DataFrame,
    risk_aversion: float,
) -> pd.Series:
    """
    Rendements d'équilibre implicites (Pi), obtenus par "optimisation
    inverse" : on suppose que les poids de marché sont DÉJÀ optimaux,
    et on retrouve mathématiquement les rendements qui justifieraient
    ce choix (c'est l'inverse de la démarche de Markowitz classique,
    qui part des rendements pour trouver les poids).

        Pi = delta * Sigma * w_marché

    C'est le point de départ neutre de Black-Litterman, bien plus stable
    que la moyenne historique brute utilisée par Markowitz.
    """
    tickers = list(cov_matrix.columns)
    weights = np.array([market_weights[t] for t in tickers])
    pi = risk_aversion * cov_matrix.values.dot(weights)
    return pd.Series(pi, index=tickers)


# ============================================================
# ÉTAPE 3 : Intégration des vues de l'investisseur
# ============================================================

def build_views_matrices(
    views: dict,
    confidences: dict,
    tickers: list,
    cov_matrix: pd.DataFrame,
    tau: float,
) -> tuple:
    """
    Construit les 3 matrices nécessaires à la formule de Black-Litterman,
    à partir de VUES ABSOLUES fournies par l'utilisateur.

    Exemple :
        views = {"NVDA": 0.40}        -> "je pense que NVDA fera 40 %/an"
        confidences = {"NVDA": 0.6}   -> confiance de 60 % dans cette vue

    Args:
        views: dict {ticker: rendement annuel attendu par l'investisseur}
        confidences: dict {ticker: confiance entre 0 (aucune) et 1 (totale)}
        tickers: liste ordonnée de tous les tickers du portefeuille
        cov_matrix: matrice de covariance annualisée
        tau: paramètre d'échelle de l'incertitude sur Pi (0.01 à 0.05
             typiquement dans la littérature)

    Returns:
        P: matrice (k vues x n actifs) — 1 sur la colonne de l'actif visé
        Q: vecteur (k,) des rendements visés
        Omega: matrice diagonale (k x k) d'incertitude de chaque vue
               (plus la confiance est haute, plus Omega est petit,
               donc plus la vue pèse fort dans le résultat final)
    """
    view_tickers = list(views.keys())
    k = len(view_tickers)
    n = len(tickers)

    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, ticker in enumerate(view_tickers):
        j = tickers.index(ticker)
        P[i, j] = 1.0
        Q[i] = views[ticker]

        confidence = confidences.get(ticker, 0.5)
        confidence = min(max(confidence, 1e-4), 0.999999)  # évite division par 0

        # Incertitude de la vue (méthode inspirée d'Idzorek, 2005) :
        # plus la confiance est faible, plus omega est grand (la vue
        # pèse moins face aux rendements d'équilibre du marché).
        view_variance = float(P[i] @ (tau * cov_matrix.values) @ P[i].T)
        omega_diag[i] = view_variance * (1.0 / confidence - 1.0)

    Omega = np.diag(omega_diag)
    return P, Q, Omega


def negative_quadratic_utility(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float,
) -> float:
    """
    Oppose de l'utilité moyenne-variance classique :
        U(w) = w'.mu - (delta/2).w'.Sigma.w

    C'est l'objectif ORIGINAL de Black & Litterman (1992), à ne pas
    confondre avec la maximisation du ratio de Sharpe utilisée dans
    markowitz.py. La différence est cruciale : la reverse optimization
    de l'étape 2 (Pi = delta * Sigma * w_marché) est l'exacte condition
    du premier ordre de CETTE utilité quadratique, pas de celle du
    ratio de Sharpe (qui soustrait risk_free_rate).

    Conséquence si on utilise le ratio de Sharpe à la place (comme le
    fait Markowitz) : dès que risk_free_rate != 0, le terme -rf déforme
    la direction du portefeuille optimal, et Black-Litterman SANS vue
    ne retombe plus sur les poids de marché — violation de la propriété
    fondamentale du modèle (vérifié empiriquement, voir
    tests/test_black_litterman.py::test_no_views_close_to_market_weights,
    et rapport chapitre 4).
    """
    ret = portfolio_return(weights, mean_returns)
    variance = portfolio_volatility(weights, cov_matrix) ** 2
    utility = ret - (risk_aversion / 2.0) * variance
    return -utility


def black_litterman_posterior(
    pi: pd.Series,
    cov_matrix: pd.DataFrame,
    tau: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple:
    """
    Combine les rendements d'équilibre (Pi) avec les vues de l'investisseur
    pour produire les rendements POSTÉRIEURS de Black-Litterman (formule
    bayésienne originale de Black & Litterman, 1992) :

        E[R] = [(tau*Sigma)^-1 + P^T.Omega^-1.P]^-1
               . [(tau*Sigma)^-1.Pi + P^T.Omega^-1.Q]

        Sigma_posterior = Sigma + [(tau*Sigma)^-1 + P^T.Omega^-1.P]^-1

    Si aucune vue n'est fournie (P vide), retourne Pi et Sigma inchangés :
    Black-Litterman se comporte alors comme un simple portefeuille
    d'équilibre de marché (poids de capitalisation).
    """
    if P.shape[0] == 0:
        return pi, cov_matrix

    sigma = cov_matrix.values
    pi_vec = pi.values

    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(Omega)

    middle = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    posterior_mean = middle @ (tau_sigma_inv @ pi_vec + P.T @ omega_inv @ Q)
    posterior_cov = sigma + middle

    posterior_returns = pd.Series(posterior_mean, index=pi.index)
    posterior_cov_df = pd.DataFrame(
        posterior_cov, index=cov_matrix.index, columns=cov_matrix.columns
    )
    return posterior_returns, posterior_cov_df


# ============================================================
# ÉTAPE 4 : Optimisation finale (identique à Markowitz, sur les
# rendements/covariance postérieurs de Black-Litterman)
# ============================================================

def optimize_black_litterman(
    returns: pd.DataFrame,
    views: dict = None,
    confidences: dict = None,
    risk_free_rate: float = 0.02,
    tau: float = 0.05,
    market_weights: dict = None,
    end_prices: pd.Series = None,
) -> dict:
    """
    Calcule le portefeuille optimal selon le modèle de Black-Litterman.

    Args:
        returns: DataFrame des rendements journaliers (colonnes = tickers)
        views: dict optionnel {ticker: rendement annuel attendu}, ex.
               {"NVDA": 0.40}. Si None ou vide, le résultat est simplement
               le portefeuille d'équilibre du marché (poids de
               capitalisation), sans ajustement.
        confidences: dict optionnel {ticker: confiance entre 0 et 1} pour
                     chaque vue (défaut 0.5 si non précisé pour une vue donnée)
        risk_free_rate: taux sans risque annuel
        tau: paramètre d'échelle de l'incertitude sur Pi (valeur standard
             dans la littérature : entre 0.01 et 0.05)
        market_weights: dict optionnel {ticker: poids}. Si non fourni,
                        récupéré automatiquement via capitalisation
                        boursière (yfinance).
        end_prices: pd.Series {ticker: prix de clôture à la fin de la
                    période historique analysée}. IMPORTANT pour la
                    cohérence temporelle avec Markowitz : permet de
                    calculer les poids de marché à la fin de la période
                    étudiée plutôt qu'avec la capitalisation d'aujourd'hui
                    (voir get_market_cap_weights). Ignoré si
                    market_weights est fourni directement.

    Returns:
        dict avec :
            - 'weights': poids optimaux par ticker
            - 'expected_return', 'volatility', 'sharpe_ratio': performance
              du portefeuille final
            - 'market_weights': poids de marché utilisés comme ancrage
            - 'equilibrium_returns': rendements Pi, avant prise en compte des vues
            - 'posterior_returns': rendements après prise en compte des vues
              (identiques à equilibrium_returns si aucune vue fournie)
    """
    tickers = list(returns.columns)
    n_assets = len(tickers)
    views = views or {}
    confidences = confidences or {}

    if n_assets == 0:
        raise ValueError("Aucun actif fourni pour l'optimisation.")

    # Étape 1 : poids de marché, alignés sur la fin de la période historique
    if market_weights is None:
        market_weights = get_market_cap_weights(tickers, end_prices=end_prices)

    # Rendements/covariance historiques (nécessaires pour estimer delta
    # et pour la matrice de covariance utilisée dans tout le modèle)
    mean_returns = compute_mean_returns(returns, annualize=True)
    cov_matrix = compute_covariance_matrix(returns, annualize=True)

    # Étape 2 : rendements d'équilibre implicites
    risk_aversion = estimate_risk_aversion(
        market_weights, mean_returns, cov_matrix, risk_free_rate
    )
    pi = compute_implied_equilibrium_returns(market_weights, cov_matrix, risk_aversion)

    # Étape 3 : intégration des vues (si fournies)
    P, Q, Omega = build_views_matrices(views, confidences, tickers, cov_matrix, tau)
    posterior_returns, posterior_cov = black_litterman_posterior(
        pi, cov_matrix, tau, P, Q, Omega
    )

    # Étape 4 : optimisation d'utilité moyenne-variance (PAS max Sharpe),
    # sur les rendements/covariance postérieurs.
    #
    # IMPORTANT : contrairement à Markowitz (markowitz.py), qui maximise
    # le ratio de Sharpe, Black-Litterman doit maximiser l'utilité
    # quadratique w'.mu - (delta/2).w'.Sigma.w, avec le MÊME delta que
    # celui utilisé à l'étape 2 pour calculer Pi. C'est la seule façon
    # de garantir la propriété d'équilibre : sans vue (posterior = Pi),
    # l'optimum retombe exactement sur les poids de marché, car Pi est
    # justement construit comme la condition du premier ordre de CETTE
    # utilité (reverse optimization). Utiliser le ratio de Sharpe ici
    # introduirait un terme -risk_free_rate qui casse cette cohérence
    # (voir negative_quadratic_utility ci-dessus).
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    constraints = build_constraints(n_assets)
    bounds = build_bounds(n_assets, max_weight=1.0)

    result = minimize(
        fun=negative_quadratic_utility,
        x0=initial_weights,
        args=(posterior_returns, posterior_cov, risk_aversion),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"L'optimisation Black-Litterman a échoué : {result.message}")

    optimal_weights = np.where(np.abs(result.x) < 1e-6, 0.0, result.x)

    final_return = portfolio_return(optimal_weights, posterior_returns)
    final_volatility = portfolio_volatility(optimal_weights, posterior_cov)
    final_sharpe = portfolio_sharpe_ratio(
        optimal_weights, posterior_returns, posterior_cov, risk_free_rate
    )

    # Avertissement : capitalisation approximative pour les entreprises à
    # classes d'actions multiples (biais confirmé empiriquement sur GOOGL,
    # voir docstring de get_market_cap_weights)
    affected = sorted(set(tickers) & MULTI_CLASS_TICKERS)
    warnings = []
    if affected:
        warnings.append(
            f"Capitalisation approximative pour {', '.join(affected)} : "
            "entreprise(s) à classes d'actions multiples, le nombre "
            "d'actions récupéré via yfinance peut ne couvrir qu'une seule "
            "classe, sous-estimant leur poids de marché réel."
        )

    return {
        "weights": {
            ticker: round(float(weight), 4)
            for ticker, weight in zip(tickers, optimal_weights)
        },
        "expected_return": round(final_return, 4),
        "volatility": round(final_volatility, 4),
        "sharpe_ratio": round(final_sharpe, 4),
        "market_weights": {t: round(w, 4) for t, w in market_weights.items()},
        "equilibrium_returns": {t: round(float(v), 4) for t, v in pi.items()},
        "posterior_returns": {
            t: round(float(v), 4) for t, v in posterior_returns.items()
        },
        "warnings": warnings,
    }