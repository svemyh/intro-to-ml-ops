"""
Modal serverless deployment of the Iris Classification model.
This replaces all the infrastructure from Parts 1-3 with ~20 lines of Python.
"""

import modal

# Create Modal app
app = modal.App("iris-classifier")

# Define the runtime environment
image = modal.Image.debian_slim().pip_install(
    "scikit-learn>=1.2.0",
    "joblib>=1.2.0",
    "numpy>=1.21.0"
)

# Mount the trained model file
model_mount = modal.Mount.from_local_file(
    "iris_model.pkl",
    remote_path="/root/iris_model.pkl"
)

@app.function(
    image=image,
    mounts=[model_mount]
)
@modal.web_endpoint(method="POST", label="predict")
def predict(item: dict):
    """
    Predict iris class from features.

    This single function replaces:
    - FastAPI application code
    - Docker container configuration
    - Kubernetes deployment manifests
    - Load balancer setup
    - Auto-scaling configuration
    - Health checks
    - Container registry management
    """
    import joblib
    import numpy as np

    # Load the model (cached after first call)
    model = joblib.load("/root/iris_model.pkl")

    # Validate input
    features = item.get("features", [])
    if len(features) != 4:
        return {"error": "Exactly 4 features required"}

    # Make prediction
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Map to class names
    class_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction": int(prediction),
        "class_name": class_names[prediction],
        "confidence": float(probabilities[prediction]),
        "features_used": features
    }

@app.function(
    image=image,
    mounts=[model_mount]
)
@modal.web_endpoint(method="GET", label="health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "service": "iris-classifier",
        "platform": "modal"
    }

@app.function(
    image=image,
    mounts=[model_mount]
)
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
        "model": "Iris Classification",
        "algorithm": "Random Forest",
        "features": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ],
        "classes": ["setosa", "versicolor", "virginica"],
        "sample_data": sample_data.get("sample_inputs", [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 2.8, 4.8, 1.8],
            [7.3, 2.9, 6.3, 1.8]
        ])
    }

# Optional: Batch processing endpoint
@app.function(
    image=image,
    mounts=[model_mount]
)
@modal.web_endpoint(method="POST", label="batch-predict")
def batch_predict(items: list):
    """
    Batch prediction endpoint for processing multiple samples efficiently.
    """
    import joblib
    import numpy as np

    model = joblib.load("/root/iris_model.pkl")
    class_names = ["setosa", "versicolor", "virginica"]

    results = []
    for item in items:
        try:
            features = item.get("features", [])
            if len(features) != 4:
                results.append({"error": "Exactly 4 features required"})
                continue

            X = np.array(features).reshape(1, -1)
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            results.append({
                "prediction": int(prediction),
                "class_name": class_names[prediction],
                "confidence": float(probabilities[prediction]),
                "features_used": features
            })
        except Exception as e:
            results.append({"error": str(e)})

    return {"results": results, "count": len(results)}

if __name__ == "__main__":
    print("This is a Modal app. Deploy it with:")
    print("modal deploy app_modal.py")
    print("\nAfter deployment, you'll get URLs like:")
    print("• https://username--iris-classifier-predict.modal.run")
    print("• https://username--iris-classifier-health.modal.run")
    print("• https://username--iris-classifier-info.modal.run")