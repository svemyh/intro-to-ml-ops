#!/usr/bin/env python3
"""
Create a sample trained ML model for the MLOps workshop.
This script trains a simple classifier on the Iris dataset and saves it.
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_sample_model():
    """Create and save a sample ML model using the Iris dataset."""

    print("🌸 Loading Iris dataset...")
    # Load the famous Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {iris.feature_names}")
    print(f"Target names: {iris.target_names}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train a Random Forest classifier
    print("\n🚀 Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=3
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model trained successfully!")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Feature importances: {model.feature_importances_.round(3)}")

    # Show classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Save the model
    model_filename = 'iris_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\n💾 Model saved as '{model_filename}'")

    # Test loading the model
    loaded_model = joblib.load(model_filename)
    test_prediction = loaded_model.predict([[5.1, 3.5, 1.4, 0.2]])
    predicted_class = iris.target_names[test_prediction[0]]

    print(f"\n🧪 Test prediction for [5.1, 3.5, 1.4, 0.2]: {predicted_class}")

    # Create a sample data file for testing
    sample_data = {
        'feature_names': list(iris.feature_names),
        'target_names': list(iris.target_names),
        'sample_inputs': [
            [5.1, 3.5, 1.4, 0.2],  # setosa
            [6.2, 2.8, 4.8, 1.8],  # versicolor
            [7.3, 2.9, 6.3, 1.8],  # virginica
        ]
    }

    import json
    with open('sample_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"\n📝 Sample data saved as 'sample_data.json'")
    print("\n🎉 Setup complete! You're ready for Part 1.")

if __name__ == "__main__":
    create_sample_model()