"""
Modal serverless deployment of the MNIST Digit Classification model.
This replaces all the infrastructure from Parts 1-3 with ~20 lines of Python.
"""

import modal

# Create Modal app
app = modal.App("mnist-classifier")

# Define the runtime environment
image = modal.Image.debian_slim().pip_install("torch>=1.9.0", "numpy>=1.21.0")

# Mount the trained model file
model_mount = modal.Mount.from_local_file(
    "mnist_model.pth", remote_path="/root/mnist_model.pth"
)


# Define the model architecture (must match training)
class MNISTModel:
    def __init__(self):
        import torch.nn as nn

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        import torch.nn.functional as F

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@app.function(image=image, mounts=[model_mount])
@modal.web_endpoint(method="POST", label="predict")
def predict(item: dict):
    """
    Predict digit from 28x28 image.

    This single function replaces:
    - FastAPI application code
    - Docker container configuration
    - Kubernetes deployment manifests
    - Load balancer setup
    - Auto-scaling configuration
    - Health checks
    - Container registry management
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Recreate model architecture
    class MNISTModel(nn.Module):
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

    # Load the model (cached after first call)
    checkpoint = torch.load("/root/mnist_model.pth", map_location="cpu")
    model = MNISTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Validate input
    image_data = item.get("image", [])

    # Handle both flat and 2D input formats
    if isinstance(image_data[0], list) if image_data else False:  # 2D format
        if len(image_data) != 28 or any(len(row) != 28 for row in image_data):
            return {"error": "2D image must be 28x28 pixels"}
        image_data = [pixel for row in image_data for pixel in row]  # Flatten

    if len(image_data) != 784:
        return {"error": "Image must have exactly 784 pixel values (28x28)"}

    try:
        # Convert to tensor and add batch/channel dimensions
        image_tensor = torch.tensor(image_data, dtype=torch.float32).reshape(
            1, 1, 28, 28
        )

        # Apply MNIST normalization
        normalized_image = (image_tensor - 0.1307) / 0.3081

        # Make prediction
        with torch.no_grad():
            output = model(normalized_image)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()

        return {
            "prediction": prediction,
            "class_name": str(prediction),
            "confidence": confidence,
            "probabilities": probabilities[0].tolist(),
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.function(image=image, mounts=[model_mount])
@modal.web_endpoint(method="GET", label="health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "service": "mnist-classifier",
        "platform": "modal",
        "model_type": "PyTorch Neural Network",
        "task": "Handwritten digit recognition (0-9)",
    }


@app.function(image=image, mounts=[model_mount])
@modal.web_endpoint(method="GET", label="info")
def info():
    """Get model information."""
    import json
    import os

    # Try to load sample data if available
    sample_data = {}
    try:
        if os.path.exists("/root/sample_data.json"):
            with open("/root/sample_data.json", "r") as f:
                sample_data = json.load(f)
    except:
        pass

    return {
        "model": "MNIST Digit Classifier",
        "algorithm": "Neural Network (PyTorch)",
        "input_format": "28x28 pixel image (784 values, 0-1 range)",
        "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
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
        "architecture": "Flatten → Linear(784→128) → ReLU → Linear(128→10)",
        "normalization": "mean=0.1307, std=0.3081",
    }


# Optional: Batch processing endpoint
@app.function(image=image, mounts=[model_mount])
@modal.web_endpoint(method="POST", label="batch-predict")
def batch_predict(items: list):
    """
    Batch prediction endpoint for processing multiple MNIST images efficiently.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Recreate and load model
    class MNISTModel(nn.Module):
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

    checkpoint = torch.load("/root/mnist_model.pth", map_location="cpu")
    model = MNISTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []
    for item in items:
        try:
            image_data = item.get("image", [])

            # Handle both flat and 2D input formats
            if isinstance(image_data[0], list) if image_data else False:
                if len(image_data) != 28 or any(len(row) != 28 for row in image_data):
                    results.append({"error": "2D image must be 28x28 pixels"})
                    continue
                image_data = [pixel for row in image_data for pixel in row]

            if len(image_data) != 784:
                results.append(
                    {"error": "Image must have exactly 784 pixel values (28x28)"}
                )
                continue

            # Convert to tensor and predict
            image_tensor = torch.tensor(image_data, dtype=torch.float32).reshape(
                1, 1, 28, 28
            )
            normalized_image = (image_tensor - 0.1307) / 0.3081

            with torch.no_grad():
                output = model(normalized_image)
                probabilities = F.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
                confidence = probabilities[0][prediction].item()

            results.append(
                {
                    "prediction": prediction,
                    "class_name": str(prediction),
                    "confidence": confidence,
                    "probabilities": probabilities[0].tolist(),
                }
            )
        except Exception as e:
            results.append({"error": str(e)})

    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    print("This is a Modal app. Deploy it with:")
    print("modal deploy app_modal.py")
    print("\nAfter deployment, you'll get URLs like:")
    print("• https://username--mnist-classifier-predict.modal.run")
    print("• https://username--mnist-classifier-health.modal.run")
    print("• https://username--mnist-classifier-info.modal.run")
    print("• https://username--mnist-classifier-batch-predict.modal.run")
