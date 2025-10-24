"""
Chatbot logic for intent classification and response generation.
Implements confidence threshold and fallback responses.
"""

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add the src directory to the path to import nltk_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nltk_utils import (
    tokenize, 
    bag_of_words, 
    load_model_data, 
    load_intents,
    get_response_patterns,
    clean_text
)

# Try to import sentiment analysis
try:
    from sentiment_analysis import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

class ChatBot:
    """
    A chatbot class that uses a trained neural network for intent classification.
    """
    
    def __init__(self, model_path="models/chatbot_model.h5", intents_path="data/intents.json", confidence_threshold=0.75, enable_sentiment=True):
        """
        Initialize the chatbot.
        
        Args:
            model_path (str): Path to the trained model
            intents_path (str): Path to the intents JSON file
            confidence_threshold (float): Minimum confidence for intent classification
            enable_sentiment (bool): Enable sentiment analysis
        """
        self.model_path = model_path
        self.intents_path = intents_path
        self.confidence_threshold = confidence_threshold
        self.enable_sentiment = enable_sentiment
        self.words = None
        self.labels = None
        self.model = None
        self.intents_data = None
        self.sentiment_analyzer = None
        
        # Load model and data
        self._load_model()
        self._load_intents()
        
        # Initialize sentiment analysis if enabled
        if self.enable_sentiment and SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = SentimentAnalyzer()
            print("✅ Sentiment analysis enabled")
        elif self.enable_sentiment and not SENTIMENT_AVAILABLE:
            print("⚠️  Sentiment analysis requested but not available. Install textblob and vaderSentiment.")
    
    def _load_model(self):
        """Load the trained model and vocabulary."""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"✅ Model loaded from {self.model_path}")
            else:
                print(f"❌ Model not found at {self.model_path}")
                print("Please train the model first by running: python src/train.py")
                return False
            
            # Load words and labels
            self.words, self.labels = load_model_data("models")
            if self.words is None or self.labels is None:
                print("❌ Failed to load vocabulary data")
                return False
            
            print(f"✅ Vocabulary loaded: {len(self.words)} words, {len(self.labels)} labels")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _load_intents(self):
        """Load intents data for response generation."""
        try:
            self.intents_data = load_intents(self.intents_path)
            if self.intents_data is None:
                print("❌ Failed to load intents data")
                return False
            print("✅ Intents data loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading intents: {e}")
            return False
    
    def predict_intent(self, message):
        """
        Predict the intent of a message.
        
        Args:
            message (str): User input message
            
        Returns:
            tuple: (predicted_intent, confidence) or (None, 0) if confidence is too low
        """
        if self.model is None or self.words is None or self.labels is None:
            return None, 0
        
        try:
            # Clean and tokenize the message
            message = clean_text(message)
            tokens = tokenize(message)
            
            # Create bag of words
            bag = bag_of_words(tokens, self.words)
            
            # Reshape for model input
            bag = bag.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(bag, verbose=0)
            
            # Get the highest confidence prediction
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                return None, confidence
            
            # Get the predicted label
            predicted_intent = self.labels[predicted_class]
            
            return predicted_intent, confidence
            
        except Exception as e:
            print(f"❌ Error predicting intent: {e}")
            return None, 0
    
    def get_response(self, message):
        """
        Get a response for a user message.
        
        Args:
            message (str): User input message
            
        Returns:
            str: Bot response
        """
        # Predict intent
        intent, confidence = self.predict_intent(message)
        
        if intent is None:
            # Fallback response for low confidence
            fallback_responses = [
                "I'm not sure I understand. Could you rephrase that?",
                "I'm having trouble understanding. Can you try asking in a different way?",
                "I'm not certain what you mean. Could you clarify or ask something else?",
                "I don't quite follow. Could you rephrase your question?",
                "I'm not sure what you're asking. Can you try asking differently?"
            ]
            return random.choice(fallback_responses)
        
        # Get responses for the predicted intent
        responses = get_response_patterns(self.intents_data, intent)
        
        if responses:
            base_response = random.choice(responses)
            
            # Enhance response with sentiment analysis if available
            if self.sentiment_analyzer and self.enable_sentiment:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(message, method="both")
                if "combined" in sentiment_result and "error" not in sentiment_result["combined"]:
                    sentiment_response = self.sentiment_analyzer.get_sentiment_response(sentiment_result["combined"])
                    # Combine base response with sentiment-aware response
                    return f"{base_response} {sentiment_response}"
            
            return base_response
        else:
            # Fallback if no responses found for the intent
            return "I understand you're asking about something, but I don't have a specific response for that."
    
    def chat(self, message):
        """
        Main chat function that processes a message and returns a response.
        
        Args:
            message (str): User input message
            
        Returns:
            dict: Response containing message, intent, confidence, and sentiment
        """
        # Get the response
        response = self.get_response(message)
        
        # Get intent and confidence for debugging
        intent, confidence = self.predict_intent(message)
        
        result = {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "threshold": self.confidence_threshold
        }
        
        # Add sentiment analysis if available
        if self.sentiment_analyzer and self.enable_sentiment:
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(message, method="both")
            result["sentiment"] = sentiment_result
        
        return result
    
    def set_confidence_threshold(self, threshold):
        """
        Set the confidence threshold for intent classification.
        
        Args:
            threshold (float): New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"✅ Confidence threshold set to {threshold}")
        else:
            print("❌ Confidence threshold must be between 0.0 and 1.0")
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "vocabulary_size": len(self.words) if self.words else 0,
            "num_labels": len(self.labels) if self.labels else 0,
            "confidence_threshold": self.confidence_threshold,
            "labels": self.labels if self.labels else []
        }

def main():
    """Test the chatbot functionality."""
    print("🤖 Testing Chatbot...")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = ChatBot()
    
    # Test messages
    test_messages = [
        "Hello!",
        "How are you?",
        "What's your name?",
        "Thank you very much",
        "Goodbye!",
        "Can you help me?",
        "What's the weather like?",
        "I don't understand this at all"
    ]
    
    print("Testing chatbot responses:")
    print("-" * 40)
    
    for message in test_messages:
        result = chatbot.chat(message)
        print(f"User: {message}")
        print(f"Bot: {result['response']}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print("-" * 40)
    
    # Print model info
    print("\nModel Information:")
    info = chatbot.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
