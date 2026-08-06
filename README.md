# OptiFolio

Plateforme web d’aide à la décision pour l’optimisation de portefeuille, combinant méthodes classiques (Markowitz, Black-Litterman) et Machine Learning (vues prédictives injectées dans Black-Litterman).

## Fonctionnalités

- Données : téléchargement historique via yfinance, rendements log, corrélations, graphiques
- Markowitz : maximisation du ratio de Sharpe (SLSQP), poids optimaux, rendement / volatilité / Sharpe
- Black-Litterman : équilibre de marché (capitalisations), vues investisseur optionnelles, postérieur bayésien
- ML + Black-Litterman : pipeline modulaire (Ridge, Random Forest, Gradient Boosting, etc.), vues automatiques, confiance calée sur le R², clip des prédictions extrêmes
- Backtest : fenêtres glissantes hors-échantillon (Markowitz vs BL vs ML), Sharpe, Sortino, CAGR, Max Drawdown
- Interface : sidebar, Accueil, Données, Optimisation (onglets)

## Installation

python -m venv venv

Linux / macOS :
source venv/bin/activate

Windows :
venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

Dépendances principales : Flask, pandas, numpy, yfinance, scipy, scikit-learn.
Optionnel : xgboost, pyarrow (cache parquet OHLCV).

## Lancement

python run.py

Application : http://127.0.0.1:5000

Parcours recommandé :
1. Accueil — présentation
2. Données — tickers + période → téléchargement
3. Optimisation — Markowitz, Black-Litterman, ML, Comparaison et backtest

## Structure du projet

portfolio-optimizer/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   ├── routes/
│   │   ├── main.py
│   │   ├── data.py
│   │   └── optimization.py
│   ├── services/
│   │   ├── data_collector.py
│   │   ├── markowitz.py
│   │   ├── black_litterman.py
│   │   ├── ml_black_litterman.py
│   │   ├── backtest.py
│   │   ├── metrics.py
│   │   └── ml/
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


