from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Accueil : présentation du projet."""
    return render_template('index.html')


@main_bp.route('/data')
def data():
    """Données : téléchargement, prétraitements et statistiques."""
    return render_template('data.html')


@main_bp.route('/dashboard')
def dashboard():
    """Optimisation : Markowitz, Black-Litterman, ML, comparaison et backtest."""
    return render_template('dashboard.html')