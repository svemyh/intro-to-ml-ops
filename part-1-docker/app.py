#!/usr/bin/env python3
"""
FastAPI application for serving the Iris classification model.
This creates a REST API endpoint for making predictions.
"""

import os
import joblib
import numpy as np
import json
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import uvicorn

# Global variable to hold the model
model = None
model_info = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and clean up on shutdown."""
    global model, model_info

    print("🚀 Starting up the ML API service...")

    # Load the trained model
    model_path = "iris_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")

    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")

    # Load model metadata
    if os.path.exists("sample_data.json"):
        with open("sample_data.json", "r") as f:
            model_info = json.load(f)
        print(f"📊 Model info loaded: {len(model_info.get('feature_names', []))} features, {len(model_info.get('target_names', []))} classes")

    yield  # This is where the application runs

    print("🛑 Shutting down the ML API service...")

# Create FastAPI application with lifecycle management
app = FastAPI(
    title="Iris Classification API",
    description="A simple API for classifying Iris flowers using a trained Random Forest model",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[float] = Field(
        ...,
        description="List of 4 features: sepal_length, sepal_width, petal_length, petal_width",
        example=[5.1, 3.5, 1.4, 0.2]
    )

    @validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Exactly 4 features are required')
        if any(f < 0 for f in v):
            raise ValueError('All features must be non-negative')
        return v

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(description="Predicted class index (0, 1, or 2)")
    class_name: str = Field(description="Human-readable class name")
    confidence: float = Field(description="Prediction confidence/probability")
    features_used: List[float] = Field(description="Features that were used for prediction")

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
        model_info=model_info
    )

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Iris Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using the trained model."""

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert features to numpy array
        X = np.array([request.features])

        # Make prediction
        prediction = model.predict(X)[0]

        # Get prediction probabilities for confidence
        probabilities = model.predict_proba(X)[0]
        confidence = float(probabilities[prediction])

        # Get class name
        class_names = model_info.get('target_names', ['setosa', 'versicolor', 'virginica'])
        class_name = class_names[prediction] if prediction < len(class_names) else f"class_{prediction}"

        return PredictionResponse(
            prediction=int(prediction),
            class_name=class_name,
            confidence=confidence,
            features_used=request.features
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    # Run the server
    print("🌸 Starting Iris Classification API...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )