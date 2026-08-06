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
