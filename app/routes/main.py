from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Page d'accueil : sélection des actifs et lancement des optimisations."""
    return render_template('index.html')


@main_bp.route('/dashboard')
def dashboard():
    """Tableau de bord : visualisation des résultats et comparaison des méthodes."""
    return render_template('dashboard.html')
