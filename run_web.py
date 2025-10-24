#!/usr/bin/env python3
"""
Web Application Entry Point for AI Chatbot
Simple script to run the Flask web application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import flask_cors
        print("✅ Flask dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the chatbot model exists."""
    model_path = Path("models/chatbot_model.h5")
    if model_path.exists():
        print("✅ Chatbot model found")
        return True
    else:
        print("❌ Chatbot model not found")
        print("Please train the model first: python src/train.py")
        return False

def main():
    """Main function to run the web application."""
    print("🌐 AI Chatbot Web Application Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found")
        print("Please run this script from the chatbot directory")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check model
    if not check_model():
        return 1
    
    print("🚀 Starting web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Set environment variables for better development experience
    os.environ['FLASK_DEBUG'] = 'True'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Import and run the Flask app
        from app import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 Web application stopped")
        return 0
    except Exception as e:
        print(f"❌ Error starting web application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

