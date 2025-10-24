"""
Configuration file for the Flask web application.
Contains production-ready settings and environment variables.
"""

import os
from pathlib import Path

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
    
    # Application settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Chatbot settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/chatbot_model.h5')
    INTENTS_PATH = os.environ.get('INTENTS_PATH', 'data/intents.json')
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.75'))
    ENABLE_SENTIMENT = os.environ.get('ENABLE_SENTIMENT', 'True').lower() == 'true'
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'chatbot.log')
    
    # Security settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        errors = []
        
        # Check if model exists
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"Model file not found: {cls.MODEL_PATH}")
        
        # Check if intents file exists
        if not Path(cls.INTENTS_PATH).exists():
            errors.append(f"Intents file not found: {cls.INTENTS_PATH}")
        
        # Validate confidence threshold
        if not 0.0 <= cls.CONFIDENCE_THRESHOLD <= 1.0:
            errors.append(f"Confidence threshold must be between 0.0 and 1.0, got: {cls.CONFIDENCE_THRESHOLD}")
        
        # Validate port
        if not 1 <= cls.PORT <= 65535:
            errors.append(f"Port must be between 1 and 65535, got: {cls.PORT}")
        
        return errors

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production security settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY must be set in production")

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    PORT = 5001

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

