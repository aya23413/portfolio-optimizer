from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from app.config import config

db = SQLAlchemy()
migrate = Migrate()


def create_app(config_name='default'):
    """Application factory : crée et configure l'instance Flask."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Initialisation des extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Enregistrement des blueprints
    from app.routes.main import main_bp
    from app.routes.data import data_bp
    from app.routes.optimization import optimization_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(optimization_bp, url_prefix='/api/optimization')

    return app
