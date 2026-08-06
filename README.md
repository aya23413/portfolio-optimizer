# POptiFolio

Plateforme web d'aide à la décision pour l'optimisation de portefeuille,
combinant méthodes classiques (Markowitz, Black-Litterman) et techniques
de Machine Learning.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Lancement

```bash
python run.py
```

L'application est accessible sur http://127.0.0.1:5000

## Structure du projet

```
portfolio_optimizer/
├── app/
│   ├── __init__.py          # Application factory
│   ├── config.py            # Configuration (dev/test/prod)
│   ├── models/               # Modèles SQLAlchemy
│   ├── routes/                # Blueprints (vues + API)
│   │   ├── main.py            # Pages web
│   │   ├── data.py            # API de collecte de données
│   │   └── optimization.py    # API d'exécution des optimisations
│   ├── services/               # Logique métier
│   │   ├── data_collector.py   # Récupération/nettoyage des données
│   │   ├── markowitz.py        # Optimisation Markowitz
│   │   ├── black_litterman.py  # Optimisation Black-Litterman
│   │   ├── ml_optimizer.py     # Optimisation par Machine Learning
│   │   └── metrics.py          # Indicateurs de performance
│   ├── templates/               # Vues Jinja2
│   ├── static/                  # CSS / JS
│   └── utils/
├── data/
│   ├── raw/                     # Données brutes téléchargées
│   └── processed/               # Données nettoyées/prêtes à l'emploi
├── tests/
├── notebooks/                   # Exploration / prototypage
├── requirements.txt
├── run.py
└── .env.example
```

## Prochaines étapes

1. Implémenter `data_collector.fetch_historical_data` (ex. avec `yfinance`).
2. Implémenter le calcul réel dans `markowitz.py` (via `scipy.optimize` ou `PyPortfolioOpt`).
3. Implémenter `black_litterman.py`.
4. Choisir et implémenter la technique de ML dans `ml_optimizer.py`.
5. Compléter `metrics.py` pour le calcul du rendement, de la volatilité et du ratio de Sharpe.
6. Enrichir les templates avec des graphiques (Chart.js déjà inclus dans `dashboard.html`).
