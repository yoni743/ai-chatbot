"""
Flask Web Application for the AI Chatbot.
Provides a web interface for the chatbot with real-time chat functionality.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import chatbot components
from src.chatbot import ChatBot
from src.sentiment_analysis import SentimentAnalyzer
from config import config

# Get configuration
config_name = os.environ.get('FLASK_ENV', 'default')
app_config = config.get(config_name, config['default'])

# Validate configuration
config_errors = app_config.validate()
if config_errors:
    print("❌ Configuration errors:")
    for error in config_errors:
        print(f"  - {error}")
    sys.exit(1)

app = Flask(__name__)
app.config.from_object(app_config)
CORS(app, origins=app_config.CORS_ORIGINS)

# Configure logging
logging.basicConfig(
    level=getattr(logging, app_config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(app_config.LOG_FILE) if app_config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot = None
sentiment_analyzer = None

def initialize_chatbot():
    """Initialize the chatbot and sentiment analyzer."""
    global chatbot, sentiment_analyzer
    
    try:
        chatbot = ChatBot(enable_sentiment=True)
        if chatbot.model is None:
            return False, "Chatbot model not found. Please train the model first."
        
        # Initialize sentiment analyzer separately for web interface
        try:
            sentiment_analyzer = SentimentAnalyzer()
        except Exception as e:
            print(f"Sentiment analysis not available: {e}")
            sentiment_analyzer = None
        
        return True, "Chatbot initialized successfully"
    except Exception as e:
        return False, f"Error initializing chatbot: {str(e)}"

@app.route('/')
def index():
    """Main chat interface."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 warnings."""
    return '', 204

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages via API with robust error handling."""
    global chatbot
    
    try:
        # Check if chatbot is initialized
        if chatbot is None:
            logger.error("Chatbot not initialized")
            return jsonify({
                'error': 'Chatbot not initialized',
                'response': 'Please refresh the page and try again.'
            }), 500
        
        # Validate request
        if not request.is_json:
            logger.warning("Request is not JSON")
            return jsonify({
                'error': 'Invalid request format',
                'response': 'Please send a valid JSON request.'
            }), 400
        
        # Get and validate message
        data = request.get_json()
        if not data:
            logger.warning("Empty JSON data received")
            return jsonify({
                'error': 'Empty request data',
                'response': 'Please provide a message.'
            }), 400
        
        message = data.get('message', '').strip()
        if not message:
            logger.warning("Empty message received")
            return jsonify({
                'error': 'Empty message',
                'response': 'Please enter a message.'
            }), 400
        
        # Log the incoming message
        logger.info(f"Received message: {message[:50]}...")
        
        # Get chatbot response
        try:
            result = chatbot.chat(message)
            logger.info(f"Chatbot response generated successfully")
        except Exception as e:
            logger.error(f"Error in chatbot.chat(): {str(e)}")
            return jsonify({
                'error': 'Chatbot processing error',
                'response': 'Sorry, I encountered an error processing your message. Please try again.'
            }), 500
        
        # Validate chatbot response
        if not result or 'response' not in result:
            logger.error("Invalid chatbot response format")
            return jsonify({
                'error': 'Invalid response format',
                'response': 'Sorry, I encountered an error generating a response.'
            }), 500
        
        # Prepare response data
        response_data = {
            'response': result['response'],
            'intent': result.get('intent'),
            'confidence': float(result.get('confidence', 0)),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'sentiment': result.get('sentiment')
        }
        
        logger.info(f"Response sent successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'response': 'Sorry, I encountered an unexpected error. Please try again.'
        }), 500

@app.route('/api/status')
def status():
    """Get chatbot status and information with robust error handling."""
    global chatbot
    
    try:
        if chatbot is None:
            logger.warning("Status check: Chatbot not initialized")
            return jsonify({
                'status': 'not_initialized',
                'message': 'Chatbot not initialized'
            }), 500
        
        # Get model info safely
        try:
            info = chatbot.get_model_info()
            if not info or 'error' in info:
                logger.error(f"Error getting model info: {info}")
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to get model information'
                }), 500
        except Exception as e:
            logger.error(f"Exception getting model info: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Error retrieving model information'
            }), 500
        
        # Return successful status
        response_data = {
            'status': 'ready',
            'model_info': info,
            'sentiment_available': sentiment_analyzer is not None
        }
        
        logger.info("Status check successful")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in status endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update chatbot settings with robust error handling."""
    global chatbot
    
    try:
        if chatbot is None:
            logger.warning("Settings update: Chatbot not initialized")
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        # Validate request
        if not request.is_json:
            logger.warning("Settings request is not JSON")
            return jsonify({'error': 'Invalid request format'}), 400
        
        data = request.get_json()
        if not data:
            logger.warning("Empty settings data received")
            return jsonify({'error': 'No settings data provided'}), 400
        
        # Update confidence threshold
        if 'threshold' in data:
            try:
                threshold = float(data['threshold'])
                if 0.0 <= threshold <= 1.0:
                    chatbot.set_confidence_threshold(threshold)
                    logger.info(f"Confidence threshold updated to {threshold}")
                    return jsonify({'message': f'Confidence threshold set to {threshold}'})
                else:
                    logger.warning(f"Invalid threshold value: {threshold}")
                    return jsonify({'error': 'Threshold must be between 0.0 and 1.0'}), 400
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing threshold: {str(e)}")
                return jsonify({'error': 'Invalid threshold value'}), 400
        
        # Update sentiment analysis
        if 'sentiment' in data:
            try:
                chatbot.enable_sentiment = bool(data['sentiment'])
                logger.info(f"Sentiment analysis {'enabled' if chatbot.enable_sentiment else 'disabled'}")
                return jsonify({'message': f'Sentiment analysis {"enabled" if chatbot.enable_sentiment else "disabled"}'})
            except Exception as e:
                logger.error(f"Error updating sentiment setting: {str(e)}")
                return jsonify({'error': 'Failed to update sentiment setting'}), 500
        
        logger.warning("No valid settings provided")
        return jsonify({'error': 'No valid settings provided'}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in settings endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Clear chat history with proper error handling."""
    try:
        logger.info("Chat history cleared")
        return jsonify({'message': 'Chat cleared'})
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({'error': 'Failed to clear chat'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with proper logging."""
    logger.warning(f"404 error: {request.url}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with proper logging."""
    logger.error(f"500 error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors with proper logging."""
    logger.warning(f"400 error: {str(error)}")
    return jsonify({'error': 'Bad request'}), 400

def main():
    """Main function to run the Flask app with robust error handling."""
    print("🌐 Starting AI Chatbot Web Application...")
    print("=" * 50)
    
    try:
        # Initialize chatbot
        success, message = initialize_chatbot()
        if not success:
            print(f"❌ {message}")
            print("Please train the model first: python src/train.py")
            logger.error(f"Chatbot initialization failed: {message}")
            return 1
        
        print(f"✅ {message}")
        logger.info("Chatbot initialized successfully")
        
        # Get configuration
        host = app_config.HOST
        port = app_config.PORT
        debug_mode = app_config.DEBUG
        
        print(f"🚀 Starting web server...")
        print(f"📱 Open your browser and go to: http://localhost:{port}")
        print(f"🔧 Debug mode: {debug_mode}")
        print(f"🔧 Environment: {config_name}")
        print(f"🔧 Log level: {app_config.LOG_LEVEL}")
        print("=" * 50)
        
        # Run the Flask app
        app.run(host=host, port=port, debug=debug_mode)
        
    except KeyboardInterrupt:
        print("\n👋 Web application stopped by user")
        logger.info("Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Fatal error starting web application: {e}")
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == '__main__':
    main()
