import os

from flask import Blueprint, request, jsonify, current_app
import pandas as pd

from app.services.markowitz import optimize_markowitz
from app.services.black_litterman import optimize_black_litterman
from app.services.ml_black_litterman import optimize_ml_black_litterman
from app.services.backtest import run_backtest

optimization_bp = Blueprint('optimization', __name__)


def _load_returns(tickers: list) -> pd.DataFrame:
    """Charge le CSV de rendements déjà sauvegardé par fetch_historical_data()."""
    try:
        processed_dir = current_app.config.get(
            'DATA_PROCESSED_DIR', os.path.join('data', 'processed')
        )
    except RuntimeError:
        processed_dir = os.path.join('data', 'processed')

    suffix = "_".join(tickers)[:100]
    returns_path = os.path.join(processed_dir, f'returns_{suffix}.csv')

    if not os.path.exists(returns_path):
        raise FileNotFoundError(
            f"Aucune donnée trouvée pour {tickers}. "
            "Veuillez d'abord télécharger les données sur la page Accueil."
        )
    return pd.read_csv(returns_path, index_col=0, parse_dates=True)


def _load_full_prices(tickers: list) -> pd.DataFrame:
    """
    Charge l'historique COMPLET des prix (pas juste la dernière ligne),
    nécessaire au backtest : à chaque fenêtre glissante, Black-Litterman
    a besoin du prix de clôture à LA FIN DE CETTE FENÊTRE d'entraînement
    précise (qui change à chaque itération), pas juste à la fin de la
    période totale.
    """
    try:
        raw_dir = current_app.config.get('DATA_RAW_DIR', os.path.join('data', 'raw'))
    except RuntimeError:
        raw_dir = os.path.join('data', 'raw')

    suffix = "_".join(tickers)[:100]
    prices_path = os.path.join(raw_dir, f'prices_{suffix}.csv')

    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Fichier de prix introuvable pour {tickers}.")

    return pd.read_csv(prices_path, index_col=0, parse_dates=True)


def _load_end_prices(tickers: list) -> pd.Series:
    """
    Charge le dernier prix de clôture disponible (fin de la période
    historique téléchargée), pour aligner temporellement les poids de
    marché de Black-Litterman avec la fenêtre analysée par Markowitz.
    Retourne None si le fichier est introuvable (dégradation silencieuse,
    get_market_cap_weights bascule alors sur la capitalisation temps réel).
    """
    try:
        raw_dir = current_app.config.get('DATA_RAW_DIR', os.path.join('data', 'raw'))
    except RuntimeError:
        raw_dir = os.path.join('data', 'raw')

    suffix = "_".join(tickers)[:100]
    prices_path = os.path.join(raw_dir, f'prices_{suffix}.csv')

    if not os.path.exists(prices_path):
        return None

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    return prices.iloc[-1]


@optimization_bp.route('/markowitz', methods=['POST'])
def markowitz_route():
    """
    Calcule le portefeuille optimal de Markowitz pour une liste de tickers
    déjà téléchargée (via /api/data/fetch au préalable).

    Body JSON attendu : {"tickers": ["AAPL", "MSFT"], "risk_free_rate": 0.02}

    Recharge les rendements depuis le CSV déjà sauvegardé par
    fetch_historical_data() (pas de re-téléchargement yfinance ici).
    """
    payload = request.get_json(force=True) or {}
    tickers = payload.get('tickers', [])
    risk_free_rate = payload.get('risk_free_rate', 0.02)
    max_weight = payload.get('max_weight', 1.0)  # 1.0 = pas de contrainte (Markowitz pur)

    if not tickers:
        return jsonify({'error': 'Aucun ticker fourni'}), 400

    try:
        returns = _load_returns(tickers)
        result = optimize_markowitz(
            returns, risk_free_rate=risk_free_rate, max_weight=max_weight
        )
        return jsonify({'success': True, 'data': result})
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@optimization_bp.route('/black-litterman', methods=['POST'])
def black_litterman_route():
    """
    Calcule le portefeuille optimal de Black-Litterman.

    Body JSON attendu :
    {
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
        "risk_free_rate": 0.02,
        "tau": 0.05,
        "views": {"NVDA": 0.40},        (optionnel)
        "confidences": {"NVDA": 0.6}    (optionnel)
    }
    """
    payload = request.get_json(force=True) or {}
    tickers = payload.get('tickers', [])
    risk_free_rate = payload.get('risk_free_rate', 0.02)
    tau = payload.get('tau', 0.05)
    views = payload.get('views', {})
    confidences = payload.get('confidences', {})

    if not tickers:
        return jsonify({'error': 'Aucun ticker fourni'}), 400

    try:
        returns = _load_returns(tickers)
        end_prices = _load_end_prices(tickers)
        result = optimize_black_litterman(
            returns,
            views=views,
            confidences=confidences,
            risk_free_rate=risk_free_rate,
            tau=tau,
            end_prices=end_prices,
        )
        return jsonify({'success': True, 'data': result})
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@optimization_bp.route('/ml', methods=['POST'])
def ml_route():
    """
    Calcule le portefeuille optimal selon Hierarchical Risk Parity (HRP).

    Body JSON attendu : {"tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"], "risk_free_rate": 0.02}
    """
    payload = request.get_json(force=True) or {}
    tickers = payload.get('tickers', [])
    risk_free_rate = payload.get('risk_free_rate', 0.02)

    if not tickers:
        return jsonify({'error': 'Aucun ticker fourni'}), 400

    try:
        returns = _load_returns(tickers)
        end_prices = _load_end_prices(tickers)
        result = optimize_ml_black_litterman(
            returns, risk_free_rate=risk_free_rate, end_prices=end_prices
        )
        return jsonify({'success': True, 'data': result})
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@optimization_bp.route('/backtest', methods=['POST'])
def backtest_route():
    """
    Lance un backtest à fenêtres glissantes pour Markowitz ET
    Black-Litterman, sur exactement les mêmes fenêtres temporelles,
    pour une comparaison honnête (hors-échantillon) des deux méthodes.

    Body JSON attendu :
    {
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
        "risk_free_rate": 0.02,
        "min_train_years": 2
    }
    """
    payload = request.get_json(force=True) or {}
    tickers = payload.get('tickers', [])
    risk_free_rate = payload.get('risk_free_rate', 0.02)
    min_train_years = payload.get('min_train_years', 2)

    if not tickers:
        return jsonify({'error': 'Aucun ticker fourni'}), 400

    try:
        returns = _load_returns(tickers)
        prices = _load_full_prices(tickers)

        # Markowitz : pas de paramètre supplémentaire nécessaire par fenêtre
        markowitz_result = run_backtest(
            returns,
            method_name='Markowitz',
            optimize_fn=optimize_markowitz,
            risk_free_rate=risk_free_rate,
            min_train_years=min_train_years,
            optimize_kwargs_fn=lambda window: {'max_weight': 1.0},
        )

        # Black-Litterman : le prix de fin de période change à chaque
        # fenêtre (la fin de l'entraînement avance à chaque itération)
        def bl_kwargs(window):
            train_end_date = window['train_returns'].index.max()
            end_prices = prices.loc[:train_end_date].iloc[-1]
            return {
                'views': {},
                'confidences': {},
                'tau': 0.05,
                'end_prices': end_prices,
            }

        def ml_bl_kwargs(window):
            # optimize_ml_black_litterman génère ses propres vues en
            # interne (à partir des prédictions du modèle ML) -> pas de
            # 'views'/'confidences' à fournir ici, contrairement à
            # Black-Litterman pur (bl_kwargs ci-dessus)
            train_end_date = window['train_returns'].index.max()
            end_prices = prices.loc[:train_end_date].iloc[-1]
            return {
                'tau': 0.05,
                'end_prices': end_prices,
            }

        black_litterman_result = run_backtest(
            returns,
            method_name='Black-Litterman',
            optimize_fn=optimize_black_litterman,
            risk_free_rate=risk_free_rate,
            min_train_years=min_train_years,
            optimize_kwargs_fn=bl_kwargs,
        )

        # ML (Black-Litterman + IA) : même logique de end_prices par
        # fenêtre que Black-Litterman seul, puisque optimize_ml_black_litterman
        # repose sur le même mécanisme d'ancrage temporel
        ml_result = run_backtest(
            returns,
            method_name='ML (Black-Litterman + IA)',
            optimize_fn=optimize_ml_black_litterman,
            risk_free_rate=risk_free_rate,
            min_train_years=min_train_years,
            optimize_kwargs_fn=ml_bl_kwargs,
        )

        return jsonify({
            'success': True,
            'data': {
                'markowitz': markowitz_result,
                'black_litterman': black_litterman_result,
                'ml': ml_result,
            },
        })
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500