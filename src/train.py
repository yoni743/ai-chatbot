"""
Training script for the chatbot neural network model.
Uses TensorFlow/Keras to create a Sequential Neural Network for intent classification.
"""

import os
import sys
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nltk_utils import (
    load_intents, 
    preprocess_data, 
    create_training_data, 
    save_model_data
)

def create_model(input_shape, num_classes):
    """Create a Sequential Neural Network model for intent classification."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the neural network model."""
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


def plot_training_history(history, save_path="models/training_history.png"):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training history plot saved to {save_path}")


def evaluate_model(model, X_test, y_test, labels):
    """Evaluate the trained model."""
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test)

    print("\n📊 Model Evaluation Results:")
    print(f"Test Accuracy: {accuracy:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes, target_names=labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predicted_classes))


def main():
    """Main training function."""
    print("🤖 Starting Chatbot Training...")
    print("=" * 50)

    intents_path = "data/intents.json"
    if not os.path.exists(intents_path):
        print(f"❌ Error: Intents file not found at {intents_path}")
        print("Please make sure the data/intents.json file exists.")
        return

    print("📚 Loading training data...")
    intents_data = load_intents(intents_path)
    if intents_data is None:
        return

    print("🔧 Preprocessing data...")
    words, labels, xy = preprocess_data(intents_data)

    print("📊 Creating training data...")
    X, y = create_training_data(words, labels, xy)

    # ✅ Fixed parentheses line
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    print("🧠 Creating neural network model...")
    model = create_model(X_train.shape[1], len(labels))

    print("\nModel Architecture:")
    model.summary()

    print("\n🚀 Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    print("📈 Plotting training history...")
    plot_training_history(history)

    print("📊 Evaluating model...")
    evaluate_model(model, X_test, y_test, labels)

    print("💾 Saving model and data...")
    os.makedirs("models", exist_ok=True)
    model.save("models/chatbot_model.h5")
    print("✅ Model saved to models/chatbot_model.h5")

    save_model_data(words, labels)

    config = {
        "input_shape": X_train.shape[1],
        "num_classes": len(labels),
        "vocabulary_size": len(words),
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "test_samples": X_test.shape[0]
    }

    with open("models/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✅ Training configuration saved to models/config.json")

    final_accuracy = history.history["val_accuracy"][-1]
    print(f"\n🎉 Training completed!")
    print(f"Final validation accuracy: {final_accuracy:.4f}")
    print(f"Model saved to: models/chatbot_model.h5")
    print(f"Vocabulary size: {len(words)}")
    print(f"Number of intents: {len(labels)}")

    print("\n" + "=" * 50)
    print("✅ Chatbot training completed successfully!")
    print("You can now run the chatbot with: python src/app.py")


if __name__ == "__main__":
    main()

