#!/usr/bin/env python3
"""
Create a trained MNIST model for the MLOps workshop.
This script trains a simple neural network on MNIST and saves it for deployment.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTModel(nn.Module):
    """Simple neural network for MNIST digit classification."""

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_mnist_model():
    """Create and save a MNIST classification model."""

    print("🔢 Loading MNIST dataset...")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
        ]
    )

    # Load datasets
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    print(f"📊 Training images: {train_dataset.data.shape}")
    print(f"📊 Test images: {test_dataset.data.shape}")
    print(f"📊 Classes: {len(train_dataset.classes)} (digits 0-9)")

    # Create model
    print("\n🚀 Creating and training MNIST model...")
    model = MNISTModel()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    model.train()
    num_epochs = 3

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

            if batch_idx % 200 == 0:
                print(
                    f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        epoch_acc = 100.0 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(
            f"   ✅ Epoch {epoch+1} complete: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    # Evaluate the model
    print("\n📈 Evaluating model on test set...")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total

    print(f"✅ Test Results: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Save the model using torch.save (preserves model structure)
    print("\n💾 Saving trained model...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "MNISTModel",
            "input_shape": [28, 28],
            "num_classes": 10,
            "class_names": [str(i) for i in range(10)],
            "test_accuracy": test_accuracy,
            "normalization": {"mean": 0.1307, "std": 0.3081},
        },
        "mnist_model.pth",
    )

    # Also save just the state dict for easier loading
    torch.save(model.state_dict(), "mnist_model_weights.pth")

    # Create a simplified version for API serving (just weights as numpy arrays)
    model_data = {}
    for name, param in model.named_parameters():
        model_data[name] = param.detach().cpu().numpy()

    with open("mnist_model_weights.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"   📁 Saved 'mnist_model.pth' (complete model)")
    print(f"   📁 Saved 'mnist_model_weights.pth' (weights only)")
    print(f"   📁 Saved 'mnist_model_weights.pkl' (numpy arrays)")

    # Create sample data for API testing
    print("\n📝 Creating sample data for API testing...")

    # Get some test samples
    test_samples = []
    test_labels = []

    # Use raw MNIST data (without normalization) for API examples
    raw_test_dataset = datasets.MNIST(
        root="./data", train=False, download=False, transform=transforms.ToTensor()
    )

    # Get first few test samples
    for i in range(10):  # One example of each digit
        image, label = raw_test_dataset[i]
        image_array = (
            image.squeeze().numpy()
        )  # Remove channel dimension and convert to numpy
        test_samples.append(image_array.tolist())
        test_labels.append(int(label))

    # Create sample data file
    sample_data = {
        "model_info": {
            "name": "MNIST Digit Classifier",
            "type": "Neural Network",
            "input_shape": [28, 28],
            "classes": [str(i) for i in range(10)],
            "class_names": [
                "Zero",
                "One",
                "Two",
                "Three",
                "Four",
                "Five",
                "Six",
                "Seven",
                "Eight",
                "Nine",
            ],
            "accuracy": f"{test_accuracy:.2f}%",
            "normalization": {
                "mean": 0.1307,
                "std": 0.3081,
                "note": "Input images should be normalized: (pixel_value - mean) / std",
            },
        },
        "sample_inputs": [
            {
                "image": test_samples[i],
                "true_label": test_labels[i],
                "description": f"Sample image of digit {test_labels[i]}",
            }
            for i in range(5)  # Include 5 samples
        ],
        "api_examples": [
            {
                "endpoint": "/predict",
                "method": "POST",
                "input_format": "List of 784 pixel values (28x28 flattened)",
                "example_input": test_samples[0],  # Flattened 28x28 image
                "expected_output": {
                    "prediction": test_labels[0],
                    "class_name": str(test_labels[0]),
                    "confidence": "float between 0-1",
                },
            }
        ],
    }

    with open("sample_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"   📁 Saved 'sample_data.json'")

    # Test model loading and inference
    print("\n🧪 Testing model loading and inference...")

    # Test PyTorch loading
    checkpoint = torch.load("mnist_model.pth")
    test_model = MNISTModel()
    test_model.load_state_dict(checkpoint["model_state_dict"])
    test_model.eval()

    # Test with first sample
    test_image = (
        torch.tensor(test_samples[0]).float().unsqueeze(0).unsqueeze(0)
    )  # Add batch and channel dims

    # Apply normalization
    normalized_image = (test_image - checkpoint["normalization"]["mean"]) / checkpoint[
        "normalization"
    ]["std"]

    with torch.no_grad():
        output = test_model(normalized_image)
        probabilities = F.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    print(f"   🎯 Test prediction: {prediction} (confidence: {confidence:.3f})")
    print(f"   🎯 Expected: {test_labels[0]}")

    # Save a simple inference function for the API
    inference_code = '''
def predict_digit(image_array, model):
    """
    Predict digit from 28x28 image array.

    Args:
        image_array: 28x28 numpy array or list of 784 values
        model: Loaded PyTorch model

    Returns:
        dict: prediction, class_name, confidence
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    # Handle different input formats
    if isinstance(image_array, list):
        if len(image_array) == 784:  # Flattened
            image_array = np.array(image_array).reshape(28, 28)
        else:
            image_array = np.array(image_array)

    # Convert to tensor and add batch/channel dimensions
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Normalize (MNIST normalization)
    normalized_image = (image_tensor - 0.1307) / 0.3081

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(normalized_image)
        probabilities = F.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    return {
        'prediction': prediction,
        'class_name': str(prediction),
        'confidence': confidence,
        'probabilities': probabilities[0].tolist()
    }
'''

    with open("inference_utils.py", "w") as f:
        f.write(inference_code)

    print(f"   📁 Saved 'inference_utils.py'")

    print("\n🎉 MNIST model setup complete!")
    print("\n📋 Files created:")
    print("   • mnist_model.pth - Complete PyTorch model")
    print("   • mnist_model_weights.pth - Model weights only")
    print("   • mnist_model_weights.pkl - Numpy arrays (for non-PyTorch APIs)")
    print("   • sample_data.json - API testing data")
    print("   • inference_utils.py - Helper functions")
    print("\n🚀 Ready for MLOps deployment!")


if __name__ == "__main__":
    create_mnist_model()
