from flask import Blueprint, request, jsonify

from app.services import markowitz, black_litterman, ml_optimizer, metrics

optimization_bp = Blueprint('optimization', __name__)

METHODS = {
    'markowitz': markowitz.optimize,
    'black_litterman': black_litterman.optimize,
    'ml': ml_optimizer.optimize,
}


@optimization_bp.route('/run', methods=['POST'])
def run_optimization():
    """
    Lance une méthode d'optimisation sur un ensemble d'actifs.
    Body JSON attendu : {"method": "markowitz", "tickers": [...], "returns_data": {...}}
    """
    payload = request.get_json(force=True) or {}
    method = payload.get('method')

    if method not in METHODS:
        return jsonify({'error': f"Méthode inconnue : {method}"}), 400

    try:
        weights = METHODS[method](payload)
        perf = metrics.compute_performance(weights, payload)
        return jsonify({'success': True, 'method': method, 'weights': weights, 'performance': perf})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@optimization_bp.route('/compare', methods=['POST'])
def compare_methods():
    """Exécute toutes les méthodes disponibles et retourne un comparatif."""
    payload = request.get_json(force=True) or {}
    results = {}

    for name, func in METHODS.items():
        try:
            weights = func(payload)
            results[name] = {
                'weights': weights,
                'performance': metrics.compute_performance(weights, payload),
            }
        except Exception as exc:
            results[name] = {'error': str(exc)}

    return jsonify({'success': True, 'results': results})
