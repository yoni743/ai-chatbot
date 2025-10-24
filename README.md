# 🤖 AI Chatbot Flask Web App

A production-ready Flask web application that serves an AI chatbot with a modern web UI and REST API. Built with Python, Flask, TensorFlow/Keras, and NLTK. You can train your model locally and interact with it via a beautiful browser interface or programmatically through a JSON API.

## 🌟 Features

- **Flask Web App**: Responsive, modern chat UI served by Flask
- **REST API**: Simple `/api/chat` endpoint for programmatic access
- **Intent Classification**: TensorFlow/Keras model for robust intent recognition
- **NLTK Preprocessing**: Tokenization, stemming, and bag-of-words
- **Configurable**: Confidence threshold and optional sentiment analysis
- **Batteries Included**: Training scripts, CLI, and notebooks for model development

## 🧩 Setup

### Prerequisites

- Python 3.8+
- pip
- Git (optional, for cloning)

### Installation

```bash
# Clone and enter the project
git clone https://github.com/yoni743/ai-chatbot.git
cd chatbot

# (Recommended) Create and activate a virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration (optional)

Create a `.env` file in the project root if you need to override defaults:

```env
FLASK_ENV=development
SECRET_KEY=your-secret-key
MODEL_PATH=models/chatbot_model.h5
```

## ▶️ Usage

### Run the Web App

```bash
python run_web.py
```

Then open http://localhost:5000 in your browser.

### REST API

- **Endpoint**: `POST /api/chat`
- **Request**:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello there"}'
```

- **Response**:

```json
{
  "reply": "Hello! How can I help you today?",
  "intent": "greeting",
  "confidence": 0.93
}
```

## 📁 Project Structure

```
chatbot/
├── data/
│   └── intents.json              # Training data with intents and patterns
├── models/
│   ├── chatbot_model.h5         # Trained TensorFlow model
│   ├── words.pkl                # Vocabulary data
│   ├── labels.pkl               # Intent labels
│   └── config.json              # Model configuration
├── src/
│   ├── train.py                 # Model training script
│   ├── chatbot.py               # Chatbot logic and classification
│   ├── nltk_utils.py            # NLTK utilities for text processing
│   ├── sentiment_analysis.py    # Sentiment analysis utilities
│   └── app.py                   # CLI chat application
├── templates/
│   └── index.html               # Web interface HTML template
├── static/
│   ├── style.css                # Web interface CSS styles
│   └── script.js                # Web interface JavaScript
├── notebooks/
│   └── training.ipynb            # Jupyter notebook for training visualization
├── app.py                       # Flask web application
├── run_web.py                   # Web application launcher
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   # If you have git
   git clone <repository-url>
   cd chatbot
   
   # Or simply navigate to the chatbot directory
   cd chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatic on first run)
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

### Training the Model

Train the chatbot model with your intents data:

```bash
python src/train.py
```

This will:
- Load and preprocess the intents data
- Create a neural network model
- Train the model with early stopping
- Save the trained model to `models/chatbot_model.h5`
- Generate training visualizations

### Running the Chatbot

#### 🌐 Web Interface (Recommended)

Launch the modern web interface:

```bash
python run_web.py
```

Then open your browser and go to: **http://localhost:5000**

Features:
- Beautiful, responsive web interface
- Real-time chat with the AI
- Settings panel for configuration
- Debug mode for technical details
- Sentiment analysis display
- Mobile-friendly design

#### 💻 Command Line Interface

For terminal-based interaction:

```bash
python src/app.py
```

## 💬 Example Conversation

```
🤖 AI Chatbot Prototype
╔══════════════════════════════════════════════════════════════╗
║                    🤖 AI Chatbot Prototype                   ║
║                                                              ║
║  Built with Python, TensorFlow, and NLTK                    ║
║  Type 'quit' to exit, 'help' for commands                    ║
╚══════════════════════════════════════════════════════════════╝

🎉 Welcome to the AI Chatbot!
Type 'help' for available commands or start chatting!

You: Hello!
Bot: Hello! How can I help you today?

You: What's your name?
Bot: I'm an AI chatbot designed to help and chat with you! You can call me ChatBot or just Bot.

You: How are you?
Bot: I'm doing great, thank you for asking! I'm here and ready to help you with whatever you need.

You: Thank you
Bot: You're welcome!

You: quit
👋 Thanks for chatting! Session ended.
```

## 🛠️ Available Commands

### Web Interface Commands
- **Settings Panel**: Click the gear icon to access:
  - Confidence threshold adjustment
  - Sentiment analysis toggle
  - Debug mode for technical details
- **Clear Chat**: Click the trash icon to clear conversation history
- **Real-time Status**: Connection status indicator in the header

### CLI Commands
When using the command line interface, you can use these commands:

- `help` - Show available commands
- `info` - Display model information
- `threshold <value>` - Set confidence threshold (0.0-1.0)
- `sentiment on/off` - Enable/disable sentiment analysis
- `clear` - Clear the screen
- `time` - Show current time
- `stats` - Show session statistics
- `quit`, `exit`, `bye` - Exit the chat

## 📊 Training Data

The chatbot is trained on the following intents:

| Intent | Description | Sample Patterns |
|--------|-------------|----------------|
| `greeting` | Welcome messages | "Hello", "Hi", "Good morning" |
| `goodbye` | Farewell messages | "Bye", "See you later", "Goodbye" |
| `thanks` | Gratitude expressions | "Thank you", "Thanks", "Much appreciated" |
| `help` | Assistance requests | "Help", "Can you help me", "What can you do" |
| `weather` | Weather inquiries | "What's the weather", "Is it sunny" |
| `name` | Identity questions | "What's your name", "Who are you" |
| `mood` | Well-being checks | "How are you", "How do you feel" |
| `fallback` | Low confidence responses | "I don't understand", "What do you mean" |

## 🧠 Model Architecture

The chatbot uses a Sequential Neural Network with the following architecture:

```
Input Layer (vocabulary_size)
    ↓
Dense Layer (128 neurons, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Dense Layer (64 neurons, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 neurons, ReLU activation)
    ↓
Dropout (0.2)
    ↓
Output Layer (num_intents, Softmax activation)
```

## 📈 Training Process

1. **Data Preprocessing**: Tokenization, stemming, and bag-of-words conversion
2. **Model Creation**: Sequential neural network with dropout for regularization
3. **Training**: Adam optimizer with early stopping and learning rate reduction
4. **Validation**: 30% of data used for validation during training
5. **Testing**: 15% of data reserved for final model evaluation

## 🔧 Configuration

### Confidence Threshold

The chatbot uses a confidence threshold to determine when to provide fallback responses:

- **Default**: 0.75 (75% confidence required)
- **Adjustable**: Use `threshold <value>` command during chat
- **Range**: 0.0 to 1.0

### Model Parameters

Key training parameters can be adjusted in `src/train.py`:

```python
# Training configuration
epochs = 100
batch_size = 32
confidence_threshold = 0.75
```

## 📓 Jupyter Notebook

The `notebooks/training.ipynb` notebook provides:

- **Data Exploration**: Visualize intents and vocabulary
- **Model Architecture**: Interactive model visualization
- **Training Progress**: Real-time training metrics
- **Performance Analysis**: Confusion matrix and classification reports
- **Interactive Testing**: Test the model with custom inputs

To run the notebook:

```bash
jupyter notebook notebooks/training.ipynb
```

## 🎯 Advanced Features

### Sentiment Analysis (Optional)

The chatbot can be enhanced with sentiment analysis:

```python
# Add to requirements.txt
textblob>=0.17.1
vaderSentiment>=3.3.2

# Example usage in chatbot.py
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```

### Custom Intents

To add new intents, edit `data/intents.json`:

```json
{
  "tag": "custom_intent",
  "patterns": [
    "Your pattern here",
    "Another pattern"
  ],
  "responses": [
    "Your response here",
    "Alternative response"
  ]
}
```

Then retrain the model:

```bash
python src/train.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   ```
   ❌ Model not found at models/chatbot_model.h5
   ```
   **Solution**: Train the model first with `python src/train.py`

2. **NLTK data download issues**
   ```
   Error downloading NLTK data
   ```
   **Solution**: Manually download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **TensorFlow compatibility issues**
   ```
   TensorFlow version compatibility
   ```
   **Solution**: Ensure TensorFlow 2.13+ is installed:
   ```bash
   pip install tensorflow>=2.13.0
   ```

### Performance Tips

- **Increase training data**: Add more patterns per intent
- **Adjust confidence threshold**: Lower for more responses, higher for accuracy
- **Model tuning**: Experiment with different architectures in `train.py`
- **Data quality**: Ensure diverse and representative training patterns

## 📚 Dependencies

### Core Dependencies
- `tensorflow>=2.13.0` - Neural network framework
- `nltk>=3.8.1` - Natural language processing
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning utilities

### Visualization
- `matplotlib>=3.7.0` - Plotting library
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.15.0` - Interactive plots

### Optional Enhancements
- `textblob>=0.17.1` - Sentiment analysis
- `vaderSentiment>=3.3.2` - Advanced sentiment analysis
- `spacy>=3.6.0` - Advanced NLP processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent deep learning framework
- **NLTK Community** for comprehensive natural language processing tools
- **Scikit-learn Team** for machine learning utilities
- **Matplotlib/Seaborn Teams** for visualization capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the example conversation patterns
3. Ensure all dependencies are properly installed
4. Verify the model has been trained successfully

---

**Happy Chatting! 🤖💬**
