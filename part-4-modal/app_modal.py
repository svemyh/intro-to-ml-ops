"""
Modal serverless deployment of the MNIST Digit Classification model.
This replaces all the infrastructure from Parts 1-3 with ~20 lines of Python.
"""

import os
import webbrowser

import modal

# Create Modal app
app = modal.App("mnist-classifier")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "mnist_model.pth")

# Define the runtime environment
image = (
    modal.Image.debian_slim()
    .pip_install("torch>=1.9.0", "numpy>=1.21.0")
    .pip_install("modal")
    .pip_install("pydantic")
    .pip_install("torchvision")
    .pip_install("fastapi")
    .pip_install("uvicorn")
    .pip_install("python-multipart")
    .pip_install("web-browser")
    .add_local_file(model_path, "/root/mnist_model.pth", copy=True)
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


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    min_containers=1,
    timeout=60 * 60,  # [seconds]
)
@modal.asgi_app()
def start():
    import fastapi
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    fast_api_app = FastAPI()
    origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:8002",
        "https://talk.iwy.ai",
        "https://app.iwy.ai",
    ]
    fast_api_app.add_middleware(
        CORSMiddleware,
        # allow_origins=origins,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    checkpoint = torch.load(model_path, map_location="cpu")
    model = MNISTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    @fast_api_app.post("/predict")
    def predict(item: dict):
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

    return fast_api_app


if __name__ == "__main__":
    print("This is a Modal app. Deploy it with:")
    print("modal serve app_modal.py")
    print("You'll get a publically exposed URL on the form:")
    print("• https://username--predict-dev.modal.run")
