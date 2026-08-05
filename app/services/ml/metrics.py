"""
Métriques de performance de portefeuille.

Centralise ici les fonctions de calcul de métriques utilisées à travers
le projet (ratio de Sharpe, Sortino, Max Drawdown...), pour respecter la
structure de fichiers prévue initialement (app/services/metrics.py).

Les fonctions de PERFORMANCE HISTORIQUE ex-ante (rendement/volatilité/
Sharpe d'un portefeuille à partir de poids donnés) vivent dans
markowitz.py, car elles sont utilisées comme briques de l'optimisation
elle-même (fonction objectif de scipy.optimize) — les dupliquer ici
créerait une source de vérité divergente entre l'optimisation et le
reporting.

Ce module regroupe donc spécifiquement les métriques de RISQUE et
d'ÉVALUATION HORS-ÉCHANTILLON (Sortino, Max Drawdown), utilisées par le
moteur de backtest (backtest.py) pour juger les méthodes sur des
données réellement réalisées, plutôt que sur des estimations ex-ante.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def compute_max_drawdown(daily_portfolio_returns: np.ndarray) -> float:
    """
    Perte maximale (en %) qu'aurait subie un investisseur entre un pic et
    le creux suivant, sur la période testée — la métrique de risque la
    plus parlante pour un investisseur non-technique ("combien pourrais-je
    perdre au pire moment ?").

        drawdown(t) = valeur(t) / max(valeur(0..t)) - 1
        max_drawdown = min(drawdown) sur toute la période

    IMPORTANT : la courbe de richesse inclut un point d'ancrage à t=0
    (valeur = 1, avant tout rendement). Sans ce point, si la pire perte
    survient dès le tout premier jour de la période testée, elle n'est
    jamais détectée : le pic glissant (running max) démarrerait déjà
    APRÈS la chute, donnant un drawdown de 0% au lieu de la perte réelle
    (bug vérifié empiriquement : cf. tests/test_metrics.py).
    """
    if len(daily_portfolio_returns) == 0:
        return 0.0
    wealth_curve = np.concatenate(([1.0], np.cumprod(1 + daily_portfolio_returns)))
    running_max = np.maximum.accumulate(wealth_curve)
    drawdowns = wealth_curve / running_max - 1
    return float(drawdowns.min())


def compute_sortino_ratio(
    daily_portfolio_returns: np.ndarray, annualized_return: float, risk_free_rate: float
) -> float:
    """
    Variante du ratio de Sharpe qui ne pénalise que la volatilité À LA
    BAISSE (les mouvements positifs ne sont jamais considérés comme un
    "risque"). Plus pertinent que le Sharpe pour des investisseurs qui ne
    craignent pas la hausse, seulement la baisse.

        Sortino = (rendement - taux_sans_risque) / volatilité_baissière
    """
    downside_returns = daily_portfolio_returns[daily_portfolio_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_volatility = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_volatility == 0:
        return 0.0
    return float((annualized_return - risk_free_rate) / downside_volatility)


def compute_cagr(daily_portfolio_returns: np.ndarray) -> float:
    """
    CAGR (Compound Annual Growth Rate) : taux de croissance annuel
    composé RÉELLEMENT réalisé sur la période testée, à partir de la
    courbe de richesse cumulée — à ne pas confondre avec un rendement
    annualisé ex-ante (moyenne journalière x 252, utilisé dans
    markowitz.py) : le CAGR découle directement du chemin de rendements
    effectivement traversé (donc sensible à l'ordre des rendements,
    pas seulement à leur moyenne), ce qui en fait la mesure standard
    pour comparer des stratégies a posteriori (backtest).

        CAGR = (valeur_finale / valeur_initiale)^(252 / nb_jours) - 1
    """
    if len(daily_portfolio_returns) == 0:
        return 0.0
    wealth_curve = np.cumprod(1 + daily_portfolio_returns)
    final_value = wealth_curve[-1]
    n_days = len(daily_portfolio_returns)
    if final_value <= 0:
        # Perte totale ou portefeuille en territoire négatif : un CAGR
        # classique (racine n-ième) n'est pas défini mathématiquement
        # dans ce cas -> on retourne -100% plutôt qu'un nombre complexe/NaN.
        return -1.0
    years = n_days / TRADING_DAYS_PER_YEAR
    return float(final_value ** (1.0 / years) - 1.0)


def compute_calmar_ratio(daily_portfolio_returns: np.ndarray) -> float:
    """
    Ratio de Calmar : rendement composé annuel (CAGR) rapporté à la pire
    perte subie (Max Drawdown, en valeur absolue). Contrairement au
    Sharpe/Sortino (qui utilisent la volatilité comme mesure du risque),
    le Calmar mesure le risque par la pire perte RÉELLEMENT vécue —
    complémentaire, pas redondant : un portefeuille peut avoir une
    volatilité modérée mais un drawdown ponctuel sévère (ex. Markowitz
    concentré sur NVDA en 2022, cf. rapport chapitre 4), ce que le
    Sharpe seul ne capture pas bien.

        Calmar = CAGR / |Max Drawdown|
    """
    max_drawdown = compute_max_drawdown(daily_portfolio_returns)
    if max_drawdown == 0:
        return 0.0
    cagr = compute_cagr(daily_portfolio_returns)
    return float(cagr / abs(max_drawdown))


def compute_directional_accuracy(predicted_returns: np.ndarray, realized_returns: np.ndarray) -> float:
    """
    Taux de bonne direction : proportion des cas où le signe du rendement
    prédit correspond au signe du rendement réellement réalisé.

    Métrique complémentaire au R² : un modèle peut avoir un R² proche de
    0 (mauvaise précision sur la MAGNITUDE du rendement) tout en ayant un
    taux de bonne direction correct (bonne capacité à prédire au moins le
    SENS du mouvement, hausse ou baisse) — une information utile pour
    juger si le modèle a un intérêt pratique malgré un R² décevant.
    """
    if len(predicted_returns) == 0:
        return 0.0
    correct = np.sign(predicted_returns) == np.sign(realized_returns)
    return float(np.mean(correct))