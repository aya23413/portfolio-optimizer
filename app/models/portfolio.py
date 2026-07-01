from datetime import datetime
from app import db


class Portfolio(db.Model):
    """Représente un portefeuille créé par un utilisateur."""
    __tablename__ = 'portfolios'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    assets = db.relationship('Asset', backref='portfolio', lazy=True, cascade='all, delete-orphan')
    results = db.relationship('OptimizationResult', backref='portfolio', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Portfolio {self.name}>'


class Asset(db.Model):
    """Actif financier sélectionné dans un portefeuille (ex: AAPL, MSFT)."""
    __tablename__ = 'assets'

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(120))
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)

    def __repr__(self):
        return f'<Asset {self.ticker}>'


class OptimizationResult(db.Model):
    """Résultat d'une exécution d'optimisation (Markowitz, Black-Litterman, ML...)."""
    __tablename__ = 'optimization_results'

    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    method = db.Column(db.String(50), nullable=False)  # markowitz | black_litterman | ml
    weights = db.Column(db.JSON, nullable=False)        # {"AAPL": 0.3, "MSFT": 0.7}
    expected_return = db.Column(db.Float)
    volatility = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<OptimizationResult {self.method} - Portfolio {self.portfolio_id}>'
