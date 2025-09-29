#!/usr/bin/env python3
"""
FastAPI application for serving the MNIST digit classification model.
This creates a REST API endpoint for making predictions on handwritten digit images.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator


# Define the same model architecture as in training
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

    print("🚀 Starting up the MNIST Classification API service...")

    # Load the trained model
    model_path = "mnist_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        model = MNISTModel()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"📊 Model accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # Load model metadata
    if os.path.exists("sample_data.json"):
        with open("sample_data.json", "r") as f:
            model_info = json.load(f)
        print(f"📊 Model info loaded: {model_info['model_info']['name']}")

    yield  # This is where the application runs

    print("🛑 Shutting down the MNIST Classification API service...")


# Create FastAPI application with lifecycle management
app = FastAPI(
    title="MNIST Digit Classification API",
    description="A REST API for classifying handwritten digits (0-9) using a trained PyTorch neural network",
    version="1.0.0",
    lifespan=lifespan,
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


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_info: Dict[str, Any] = Field(description="Model metadata")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the service is running."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_info=model_info.get("model_info", {}),
    )


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "MNIST Digit Classification API",
        "version": "1.0.0",
        "model": "Neural Network (PyTorch)",
        "task": "Handwritten digit recognition (0-9)",
        "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"},
    }


def preprocess_image(image_data: List[float]) -> torch.Tensor:
    """Preprocess image data for model inference."""
    # Convert to numpy array and reshape to 28x28
    image_array = np.array(image_data, dtype=np.float32).reshape(28, 28)

    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

    # Apply MNIST normalization
    normalized_image = (image_tensor - 0.1307) / 0.3081

    return normalized_image


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a digit prediction using the trained model."""

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess the image
        processed_image = preprocess_image(request.image)

        # Make prediction
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


@app.get("/model-info")
async def model_info_endpoint():
    """Get detailed information about the model."""
    if not model_info:
        raise HTTPException(status_code=404, detail="Model info not available")

    return model_info


@app.get("/sample-data")
async def get_sample_data():
    """Get sample data for testing the API."""
    if not model_info:
        raise HTTPException(status_code=404, detail="Sample data not available")

    return {
        "samples": model_info.get("sample_inputs", [])[:3],  # Return first 3 samples
        "usage": "Use the 'image' field from any sample as input to /predict endpoint",
    }


if __name__ == "__main__":
    # Run the server
    print("🔢 Starting MNIST Digit Classification API...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
