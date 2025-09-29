#!/usr/bin/env python3
"""
Test the integration between the drawing client and MNIST API.
This simulates what the frontend JavaScript does.
"""

import json

import numpy as np
import requests


def test_client_api_integration():
    """Test the complete workflow from drawing simulation to prediction."""

    print("🧪 Testing Client → API Integration")

    # Test 1: API Health Check
    print("\n1. Testing API connectivity...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ API is healthy: {health['status']}")
            print(f"   📊 Model loaded: {health['model_loaded']}")
        else:
            print(f"   ❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to API: {e}")
        return False

    # Test 2: Simulate drawing data (like frontend canvas conversion)
    print("\n2. Testing image conversion and prediction...")

    # Create a simple digit "1" pattern (784 pixels = 28x28)
    test_image = create_test_digit_1()

    print(f"   📏 Generated test image: {len(test_image)} pixels")
    print(f"   📊 Pixel range: {min(test_image):.3f} to {max(test_image):.3f}")

    # Test 3: Make prediction
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"image": test_image},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful!")
            print(f"   🎯 Predicted digit: {result['prediction']}")
            print(f"   📊 Confidence: {result['confidence']:.3f}")
            print(f"   📈 Top 3 probabilities:")

            # Get top 3 predictions
            probs = result["probabilities"]
            top_3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
            for digit, prob in top_3:
                print(f"      {digit}: {prob:.3f} ({prob*100:.1f}%)")

        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Prediction request failed: {e}")
        return False

    # Test 4: Frontend server connectivity
    print("\n3. Testing frontend server...")
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Frontend server is running")
            print(f"   🌐 Access at: http://localhost:8080")
        else:
            print(f"   ❌ Frontend server error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Cannot connect to frontend: {e}")

    print(f"\n🎉 Integration test complete!")
    print(f"\n📋 Next steps:")
    print(f"   1. Open http://localhost:8080 in your browser")
    print(f"   2. Draw a digit in the canvas")
    print(f"   3. Wait 1.25 seconds for auto-prediction")
    print(f"   4. See the predicted digit light up!")

    return True


def create_test_digit_1():
    """Create a simple vertical line representing digit '1'."""
    # 28x28 image (784 pixels total)
    image = np.zeros((28, 28), dtype=np.float32)

    # Draw a vertical line in the middle (like digit "1")
    for y in range(5, 23):  # Vertical line from row 5 to 22
        image[y, 13] = 1.0  # Column 13 (middle)
        image[y, 14] = 1.0  # Column 14 (middle)

    # Add a small top line
    image[5, 11] = 1.0
    image[5, 12] = 1.0

    # Flatten to 784 pixels
    return image.flatten().tolist()


def create_test_digit_0():
    """Create a simple circle representing digit '0'."""
    image = np.zeros((28, 28), dtype=np.float32)

    # Draw a simple oval/circle
    center_x, center_y = 14, 14

    for y in range(28):
        for x in range(28):
            # Distance from center
            dx = x - center_x
            dy = y - center_y

            # Create oval shape
            if 6 <= ((dx**2) / 36 + (dy**2) / 64) <= 8:
                image[y, x] = 1.0

    return image.flatten().tolist()


if __name__ == "__main__":
    test_client_api_integration()
