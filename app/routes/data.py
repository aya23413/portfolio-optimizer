from flask import Blueprint, request, jsonify

from app.services.data_collector import fetch_historical_data

data_bp = Blueprint('data', __name__)


@data_bp.route('/fetch', methods=['POST'])
def fetch_data():
    payload = request.get_json(force=True) or {}
    tickers = payload.get('tickers', [])
    start = payload.get('start')
    end = payload.get('end')

    if not tickers:
        return jsonify({'error': 'Aucun ticker fourni'}), 400

    try:
        data = fetch_historical_data(tickers, start, end)
        return jsonify({'success': True, 'data': data})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500