import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(basedir, '.env'))


class Config:
    """Configuration de base, commune à tous les environnements."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'change-moi-en-production')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', f"sqlite:///{os.path.join(basedir, 'data', 'app.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Chemins de données
    DATA_RAW_DIR = os.path.join(basedir, 'data', 'raw')
    DATA_PROCESSED_DIR = os.path.join(basedir, 'data', 'processed')

    # Clés API (sources de données financières)
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')

    # Paramètres par défaut pour l'optimisation
    RISK_FREE_RATE = float(os.environ.get('RISK_FREE_RATE', 0.02))


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class ProductionConfig(Config):
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
