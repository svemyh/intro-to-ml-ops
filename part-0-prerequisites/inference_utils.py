def predict_digit(image_array, model):
    """
    Predict digit from 28x28 image array.

    Args:
        image_array: 28x28 numpy array or list of 784 values
        model: Loaded PyTorch model

    Returns:
        dict: prediction, class_name, confidence
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    # Handle different input formats
    if isinstance(image_array, list):
        if len(image_array) == 784:  # Flattened
            image_array = np.array(image_array).reshape(28, 28)
        else:
            image_array = np.array(image_array)

    # Convert to tensor and add batch/channel dimensions
    image_tensor = (
        torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )

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
        "prediction": prediction,
        "class_name": str(prediction),
        "confidence": confidence,
        "probabilities": probabilities[0].tolist(),
    }
