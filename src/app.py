"""
CLI Chat Application for the AI Chatbot.
A simple terminal-based chatbot conversation interface.
"""

import os
import sys
import time
from datetime import datetime

# Add the src directory to the path to import chatbot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import ChatBot

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback colors for terminals that support ANSI
    class Fore:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        RESET = '\033[0m'
    
    class Style:
        BRIGHT = '\033[1m'
        DIM = '\033[2m'
        RESET_ALL = '\033[0m'

class ChatApp:
    """
    CLI Chat Application for the AI Chatbot.
    """
    
    def __init__(self):
        """Initialize the chat application."""
        self.chatbot = None
        self.session_start = None
        self.message_count = 0
        
    def print_banner(self):
        """Print the application banner."""
        banner = f"""
{Fore.CYAN}{Style.BRIGHT}
╔══════════════════════════════════════════════════════════════╗
║                    🤖 AI Chatbot Prototype                   ║
║                                                              ║
║  Built with Python, TensorFlow, and NLTK                    ║
║  Type 'quit' to exit, 'help' for commands                    ║
╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
"""
        print(banner)
    
    def print_help(self):
        """Print help information."""
        help_text = f"""
{Fore.YELLOW}Available Commands:{Style.RESET_ALL}
  {Fore.GREEN}quit, exit, bye{Style.RESET_ALL}     - Exit the chat application
  {Fore.GREEN}help{Style.RESET_ALL}               - Show this help message
  {Fore.GREEN}info{Style.RESET_ALL}               - Show model information
  {Fore.GREEN}threshold <value>{Style.RESET_ALL}   - Set confidence threshold (0.0-1.0)
  {Fore.GREEN}sentiment on/off{Style.RESET_ALL}    - Enable/disable sentiment analysis
  {Fore.GREEN}clear{Style.RESET_ALL}              - Clear the screen
  {Fore.GREEN}time{Style.RESET_ALL}               - Show current time
  {Fore.GREEN}stats{Style.RESET_ALL}               - Show session statistics

{Fore.YELLOW}Tips:{Style.RESET_ALL}
  • Try asking about greetings, weather, help, or your name
  • The bot uses a confidence threshold to determine responses
  • If confidence is too low, you'll get a fallback response
  • Sentiment analysis provides emotional context to responses
"""
        print(help_text)
    
    def print_stats(self):
        """Print session statistics."""
        if self.session_start:
            duration = time.time() - self.session_start
            duration_str = f"{int(duration//60)}m {int(duration%60)}s"
        else:
            duration_str = "0s"
        
        stats = f"""
{Fore.CYAN}Session Statistics:{Style.RESET_ALL}
  Messages exchanged: {Fore.GREEN}{self.message_count}{Style.RESET_ALL}
  Session duration: {Fore.GREEN}{duration_str}{Style.RESET_ALL}
  Current time: {Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}
"""
        print(stats)
    
    def print_model_info(self):
        """Print model information."""
        if self.chatbot:
            info = self.chatbot.get_model_info()
            print(f"\n{Fore.CYAN}Model Information:{Style.RESET_ALL}")
            for key, value in info.items():
                if key != 'labels':  # Don't print all labels to avoid clutter
                    print(f"  {key}: {Fore.GREEN}{value}{Style.RESET_ALL}")
            
            if 'labels' in info and info['labels']:
                print(f"  Available intents: {Fore.GREEN}{', '.join(info['labels'])}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Chatbot not initialized{Style.RESET_ALL}")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
    
    def initialize_chatbot(self):
        """Initialize the chatbot."""
        print(f"{Fore.YELLOW}🔄 Initializing chatbot...{Style.RESET_ALL}")
        
        try:
            self.chatbot = ChatBot()
            if self.chatbot.model is None:
                print(f"{Fore.RED}❌ Failed to load chatbot model{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}💡 Make sure to train the model first: python src/train.py{Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}✅ Chatbot initialized successfully!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error initializing chatbot: {e}{Style.RESET_ALL}")
            return False
    
    def process_command(self, user_input):
        """Process special commands."""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            return 'quit'
        elif command == 'help':
            self.print_help()
            return 'continue'
        elif command == 'info':
            self.print_model_info()
            return 'continue'
        elif command == 'clear':
            self.clear_screen()
            return 'continue'
        elif command == 'time':
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Fore.CYAN}Current time: {Fore.GREEN}{current_time}{Style.RESET_ALL}")
            return 'continue'
        elif command == 'stats':
            self.print_stats()
            return 'continue'
        elif command.startswith('threshold '):
            try:
                threshold = float(command.split(' ')[1])
                if 0.0 <= threshold <= 1.0:
                    self.chatbot.set_confidence_threshold(threshold)
                    print(f"{Fore.GREEN}✅ Confidence threshold set to {threshold}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Threshold must be between 0.0 and 1.0{Style.RESET_ALL}")
            except (ValueError, IndexError):
                print(f"{Fore.RED}❌ Invalid threshold value. Use: threshold 0.75{Style.RESET_ALL}")
            return 'continue'
        elif command.startswith('sentiment '):
            sentiment_cmd = command.split(' ')[1].lower()
            if sentiment_cmd in ['on', 'enable', 'true']:
                self.chatbot.enable_sentiment = True
                print(f"{Fore.GREEN}✅ Sentiment analysis enabled{Style.RESET_ALL}")
            elif sentiment_cmd in ['off', 'disable', 'false']:
                self.chatbot.enable_sentiment = False
                print(f"{Fore.YELLOW}⚠️  Sentiment analysis disabled{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Invalid sentiment command. Use: sentiment on/off{Style.RESET_ALL}")
            return 'continue'
        else:
            return 'chat'
    
    def chat_loop(self):
        """Main chat loop."""
        self.session_start = time.time()
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{Fore.BLUE}You:{Style.RESET_ALL} ").strip()
                
                # Check for empty input
                if not user_input:
                    continue
                
                # Process commands
                command_result = self.process_command(user_input)
                
                if command_result == 'quit':
                    break
                elif command_result == 'continue':
                    continue
                elif command_result == 'chat':
                    # Get bot response
                    start_time = time.time()
                    result = self.chatbot.chat(user_input)
                    response_time = time.time() - start_time
                    
                    # Display response
                    print(f"{Fore.GREEN}Bot:{Style.RESET_ALL} {result['response']}")
                    
                    # Show debug info if confidence is low
                    if result['intent'] and result['confidence'] < 0.8:
                        print(f"{Fore.YELLOW}💡 Intent: {result['intent']} (confidence: {result['confidence']:.3f}){Style.RESET_ALL}")
                    
                    # Show sentiment analysis if available
                    if 'sentiment' in result and result['sentiment']:
                        sentiment_data = result['sentiment']
                        if 'combined' in sentiment_data and 'error' not in sentiment_data['combined']:
                            combined = sentiment_data['combined']
                            sentiment_emoji = {
                                'positive': '😊',
                                'negative': '😔',
                                'neutral': '😐'
                            }
                            emoji = sentiment_emoji.get(combined['sentiment'], '😐')
                            print(f"{Fore.CYAN}🧠 Sentiment: {combined['sentiment']} {emoji} (confidence: {combined['confidence']:.3f}){Style.RESET_ALL}")
                    
                    # Show response time for debugging
                    if response_time > 1.0:  # Only show if response took more than 1 second
                        print(f"{Fore.CYAN}⏱️  Response time: {response_time:.2f}s{Style.RESET_ALL}")
                    
                    self.message_count += 1
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            except EOFError:
                print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
                continue
    
    def run(self):
        """Run the chat application."""
        self.print_banner()
        
        # Initialize chatbot
        if not self.initialize_chatbot():
            return
        
        # Show welcome message
        print(f"{Fore.GREEN}🎉 Welcome to the AI Chatbot!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type 'help' for available commands or start chatting!{Style.RESET_ALL}")
        
        # Start chat loop
        self.chat_loop()
        
        # Show goodbye message
        print(f"\n{Fore.CYAN}👋 Thanks for chatting! Session ended.{Style.RESET_ALL}")
        if self.message_count > 0:
            self.print_stats()

def main():
    """Main function to run the chat application."""
    try:
        app = ChatApp()
        app.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
