"""
Sentiment analysis utilities for the chatbot.
Provides sentiment detection using TextBlob and VADER.
"""

import os
import sys

# Try to import sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not available. Install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  VADER not available. Install with: pip install vaderSentiment")

class SentimentAnalyzer:
    """
    Sentiment analysis class using TextBlob and VADER.
    """
    
    def __init__(self):
        """Initialize sentiment analyzers."""
        self.textblob_available = TEXTBLOB_AVAILABLE
        self.vader_available = VADER_AVAILABLE
        
        if self.vader_available:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        print(f"Sentiment Analysis Status:")
        print(f"  TextBlob: {'✅ Available' if self.textblob_available else '❌ Not available'}")
        print(f"  VADER: {'✅ Available' if self.vader_available else '❌ Not available'}")
    
    def analyze_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not self.textblob_available:
            return {"error": "TextBlob not available"}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Categorize sentiment
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment": sentiment,
                "confidence": abs(polarity)
            }
        except Exception as e:
            return {"error": f"TextBlob analysis failed: {e}"}
    
    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not self.vader_available:
            return {"error": "VADER not available"}
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu'],
                "compound": compound,
                "sentiment": sentiment,
                "confidence": abs(compound)
            }
        except Exception as e:
            return {"error": f"VADER analysis failed: {e}"}
    
    def analyze_sentiment(self, text, method="both"):
        """
        Analyze sentiment using specified method(s).
        
        Args:
            text (str): Text to analyze
            method (str): Analysis method ("textblob", "vader", or "both")
            
        Returns:
            dict: Combined sentiment analysis results
        """
        results = {"text": text, "method": method}
        
        if method in ["textblob", "both"] and self.textblob_available:
            results["textblob"] = self.analyze_textblob(text)
        
        if method in ["vader", "both"] and self.vader_available:
            results["vader"] = self.analyze_vader(text)
        
        # Combine results if both methods are available
        if method == "both" and self.textblob_available and self.vader_available:
            results["combined"] = self._combine_sentiments(results["textblob"], results["vader"])
        
        return results
    
    def _combine_sentiments(self, textblob_result, vader_result):
        """
        Combine TextBlob and VADER sentiment results.
        
        Args:
            textblob_result (dict): TextBlob analysis results
            vader_result (dict): VADER analysis results
            
        Returns:
            dict: Combined sentiment analysis
        """
        if "error" in textblob_result or "error" in vader_result:
            return {"error": "Cannot combine results due to errors"}
        
        # Weight the results (can be adjusted)
        textblob_weight = 0.4
        vader_weight = 0.6
        
        # Combine polarity scores
        combined_polarity = (
            textblob_result["polarity"] * textblob_weight +
            vader_result["compound"] * vader_weight
        )
        
        # Determine combined sentiment
        if combined_polarity > 0.1:
            sentiment = "positive"
        elif combined_polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "combined_polarity": combined_polarity,
            "sentiment": sentiment,
            "confidence": abs(combined_polarity),
            "textblob_polarity": textblob_result["polarity"],
            "vader_compound": vader_result["compound"]
        }
    
    def get_sentiment_response(self, sentiment_analysis):
        """
        Get appropriate response based on sentiment analysis.
        
        Args:
            sentiment_analysis (dict): Sentiment analysis results
            
        Returns:
            str: Appropriate response
        """
        if "error" in sentiment_analysis:
            return "I'm having trouble analyzing the sentiment of your message."
        
        sentiment = sentiment_analysis.get("sentiment", "neutral")
        confidence = sentiment_analysis.get("confidence", 0)
        
        responses = {
            "positive": [
                "I'm glad you're feeling positive! 😊",
                "That sounds great! I'm happy to hear that.",
                "Wonderful! I can sense the positive energy in your message.",
                "That's fantastic! I'm here to help make your day even better."
            ],
            "negative": [
                "I understand you might be feeling down. I'm here to help! 💙",
                "I can sense you're going through a tough time. How can I support you?",
                "I'm sorry to hear that. Let me know if there's anything I can do to help.",
                "I'm here for you. Sometimes talking helps, and I'm a good listener."
            ],
            "neutral": [
                "I'm here to help with whatever you need.",
                "How can I assist you today?",
                "I'm ready to help with any questions or concerns you have.",
                "What would you like to talk about?"
            ]
        }
        
        import random
        return random.choice(responses.get(sentiment, responses["neutral"]))

def main():
    """Test the sentiment analysis functionality."""
    print("🧠 Testing Sentiment Analysis...")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    test_messages = [
        "I'm so happy today!",
        "This is terrible, I hate it.",
        "The weather is okay, nothing special.",
        "I love this chatbot, it's amazing!",
        "I'm feeling really sad and lonely.",
        "Hello, how are you?",
        "This is the worst day ever!",
        "I'm excited about the new features!"
    ]
    
    print("\n📊 Sentiment Analysis Results:")
    print("-" * 50)
    
    for message in test_messages:
        print(f"\nMessage: '{message}'")
        
        # Analyze with both methods
        results = analyzer.analyze_sentiment(message, method="both")
        
        if "textblob" in results and "error" not in results["textblob"]:
            tb = results["textblob"]
            print(f"  TextBlob: {tb['sentiment']} (polarity: {tb['polarity']:.3f})")
        
        if "vader" in results and "error" not in results["vader"]:
            vd = results["vader"]
            print(f"  VADER: {vd['sentiment']} (compound: {vd['compound']:.3f})")
        
        if "combined" in results and "error" not in results["combined"]:
            combined = results["combined"]
            print(f"  Combined: {combined['sentiment']} (polarity: {combined['combined_polarity']:.3f})")
            
            # Get sentiment-based response
            response = analyzer.get_sentiment_response(combined)
            print(f"  Response: {response}")

if __name__ == "__main__":
    main()

