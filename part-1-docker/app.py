#!/usr/bin/env python3
"""
FastAPI application for serving the MNIST digit classification model.
This creates a REST API endpoint for making predictions on handwritten digit images.
"""

import json
import os
import webbrowser
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator


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


# Global variables to hold the model and metadata
model = None
model_info = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and clean up on shutdown."""
    global model, model_info

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "mnist_model.pth")

    checkpoint = torch.load(model_path, map_location="cpu")
    model = MNISTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"📊 Model accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")

    if os.path.exists("sample_data.json"):
        with open("sample_data.json", "r") as f:
            model_info = json.load(f)
        print(f"📊 Model info loaded: {model_info['model_info']['name']}")

    yield  # This is where the application runs


app = FastAPI(lifespan=lifespan)

# (this is a network security setting)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify this as your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request model for digit predictions."""

    image: Union[List[float], List[List[float]]] = Field(
        ...,
        description="28x28 image as flattened list of 784 pixel values or 2D array of pixel values (0-1 range)",
        example=[0.0] * 784,  # Flat black image
    )

    @validator("image")
    def validate_image(cls, v):
        if isinstance(v[0], list):  # 2D array format
            if len(v) != 28 or any(len(row) != 28 for row in v):
                raise ValueError("2D image must be 28x28 pixels")
            # Flatten the 2D array
            v = [pixel for row in v for pixel in row]

        if len(v) != 784:
            raise ValueError(
                "Flattened image must have exactly 784 pixel values (28x28)"
            )

        if any(not isinstance(pixel, (int, float)) for pixel in v):
            raise ValueError("All pixel values must be numbers")

        # Ensure values are in valid range (0-1)
        if any(pixel < 0 or pixel > 1 for pixel in v):
            raise ValueError("Pixel values must be between 0 and 1")

        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: int = Field(description="Predicted digit (0-9)")
    class_name: str = Field(description="Digit as string")
    confidence: float = Field(description="Prediction confidence/probability")
    probabilities: List[float] = Field(description="Probability for each digit 0-9")
    image_shape: List[int] = Field(description="Shape of processed image [28, 28]")


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {"message": "MNIST Digit Classification API"}


@app.get("/health")
async def health():
    """Health check endpoint for monitoring and Kubernetes probes."""
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "MNIST Digit Classification API",
    }


def preprocess_image(image_data: List[float]) -> torch.Tensor:
    """Preprocess image data for model inference."""
    image_array = np.array(image_data, dtype=np.float32).reshape(28, 28)

    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

    normalized_image = (image_tensor - 0.1307) / 0.3081  # MNIST normalization

    return normalized_image


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a digit prediction using the trained model."""
    try:
        processed_image = preprocess_image(request.image)

        with torch.no_grad():
            output = model(processed_image)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()

        return PredictionResponse(
            prediction=prediction,
            class_name=str(prediction),
            confidence=confidence,
            probabilities=probabilities[0].tolist(),
            image_shape=[28, 28],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    print("🔢 Starting MNIST Digit Classification API...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    client_html_path = os.path.join(script_dir, "..", "client", "index.html")

    if os.path.exists(client_html_path):
        client_url = f"file://{os.path.abspath(client_html_path)}"
        print("###############################################")
        print(f"🌐 Open the client interface: {client_url}")
        print("📝 Or copy this link to your browser to test the digit classifier!")
        print("###############################################")

        webbrowser.open(client_url)  # Auto-open

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
