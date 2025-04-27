from flask import Flask
from flask_cors import CORS
from config.config import Config
from database.db_connection import mongo
from controllers.test_controller import test_bp
from controllers.match_controller import match_bp

def create_app():
    app = Flask(__name__)

    # Load Config
    app.config.from_object(Config)

    # Enable CORS
    CORS(app)

    # Initialize MongoDB
    mongo.init_app(app)

    # Register Blueprints
    app.register_blueprint(test_bp, url_prefix="/api/test")
    app.register_blueprint(match_bp, url_prefix="/api")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
