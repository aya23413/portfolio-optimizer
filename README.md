# OptiFolio

Plateforme web d’aide à la décision pour l’**optimisation de portefeuille**,
combinant méthodes classiques (**Markowitz**, **Black-Litterman**) et
**Machine Learning** (vues prédictives injectées dans Black-Litterman).

---

## Fonctionnalités

| Module | Description |
|--------|-------------|
| **Données** | Téléchargement historique via `yfinance`, rendements log, corrélations, graphiques |
| **Markowitz** | Maximisation du ratio de Sharpe (SLSQP), poids optimaux, rendement / vol / Sharpe |
| **Black-Litterman** | Équilibre de marché (caps), vues investisseur optionnelles, postérieur bayésien |
| **ML + Black-Litterman** | Pipeline modulaire (Ridge, RF, Gradient Boosting, …) → vues auto, confiance calée sur le R², clip des prédictions extrêmes |
| **Backtest** | Fenêtres glissantes hors-échantillon (Markowitz vs BL vs ML), Sharpe / Sortino / CAGR / Max DD |
| **Interface** | Sidebar, Accueil, Données, Optimisation (onglets) |

---

## Installation

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
# venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

Lancement
Bashpython run.py
Application : http://127.0.0.1:5000
Parcours recommandé :

Accueil — présentation
Données — tickers + période → téléchargement
Optimisation — Markowitz · Black-Litterman · ML · Comparaison & backtest


Structure du projet
textportfolio-optimizer/
├── app/
│   ├── __init__.py              # Application factory
│   ├── config.py
│   ├── models/
│   ├── routes/
│   │   ├── main.py              # Pages : /, /data, /dashboard
│   │   ├── data.py              # API collecte de données
│   │   └── optimization.py      # API Markowitz, BL, ML, backtest
│   ├── services/
│   │   ├── data_collector.py
│   │   ├── markowitz.py
│   │   ├── black_litterman.py
│   │   ├── ml_black_litterman.py
│   │   ├── backtest.py
│   │   ├── metrics.py
│   │   └── ml/                  # Pipeline ML modulaire
│   ├── templates/
│   └── static/
│       ├── css/style.css
│       └── js/
├── data/
│   ├── raw/
│   └── processed/
├── cache/
├── tests/
├── requirements.txt
├── run.py
└── .env.example

API (aperçu)



































MéthodeRouteRôlePOST/api/data/fetchTélécharger prix / rendementsPOST/api/optimization/markowitzPortefeuille max SharpePOST/api/optimization/black-littermanBL (± vues)POST/api/optimization/mlML + BLPOST/api/optimization/backtestBacktest multi-méthodes

Méthodologie (résumé)

Markowitz — moyenne / covariance annualisées, max Sharpe (poids ≥ 0, somme = 1).
Black-Litterman — équilibre de marché + vues (manuelles ou ML).
ML — modèles par actif, diagnostics hold-out, confiance via R², clip des vues extrêmes.
Backtest — fenêtres expanding, comparaison hors-échantillon des trois méthodes.

Le faible pouvoir prédictif (R² souvent ≤ 0 sur actions) est attendu ; l’architecture bayésienne limite l’impact d’un signal bruité.

Tests
Bashpytest tests/ -q

Licence / usage
Projet académique / démonstration. Les résultats ne constituent pas un conseil en investissement.
