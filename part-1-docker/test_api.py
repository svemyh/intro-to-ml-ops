#!/usr/bin/env python3
"""
Test script for the Iris Classification API.
Run this to verify that your API is working correctly.
"""

import requests
import json
import sys
import time

def test_api(base_url="http://localhost:8000"):
    """Test all API endpoints."""

    print(f"🧪 Testing API at {base_url}")

    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed: {health_data['status']}")
            print(f"   📊 Model loaded: {health_data['model_loaded']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False

    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Root endpoint working")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Root endpoint failed: {e}")

    # Test 3: Prediction with valid data
    print("\n3. Testing prediction with valid data...")
    test_cases = [
        {
            "name": "Setosa sample",
            "features": [5.1, 3.5, 1.4, 0.2],
            "expected_class": "setosa"
        },
        {
            "name": "Versicolor sample",
            "features": [6.2, 2.8, 4.8, 1.8],
            "expected_class": "versicolor"
        },
        {
            "name": "Virginica sample",
            "features": [7.3, 2.9, 6.3, 1.8],
            "expected_class": "virginica"
        }
    ]

    for test_case in test_cases:
        try:
            payload = {"features": test_case["features"]}
            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                predicted_class = result["class_name"]
                confidence = result["confidence"]
                print(f"   ✅ {test_case['name']}: {predicted_class} (confidence: {confidence:.3f})")

                if predicted_class == test_case["expected_class"]:
                    print(f"      🎯 Prediction matches expected class!")
                else:
                    print(f"      ⚠️  Expected {test_case['expected_class']}, got {predicted_class}")
            else:
                print(f"   ❌ {test_case['name']} failed: {response.status_code}")
                print(f"      Response: {response.text}")

        except Exception as e:
            print(f"   ❌ {test_case['name']} failed: {e}")

    # Test 4: Invalid input validation
    print("\n4. Testing input validation...")
    invalid_cases = [
        {
            "name": "Too few features",
            "features": [5.1, 3.5],
            "should_fail": True
        },
        {
            "name": "Too many features",
            "features": [5.1, 3.5, 1.4, 0.2, 1.0],
            "should_fail": True
        },
        {
            "name": "Negative features",
            "features": [-1.0, 3.5, 1.4, 0.2],
            "should_fail": True
        }
    ]

    for test_case in invalid_cases:
        try:
            payload = {"features": test_case["features"]}
            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if test_case["should_fail"] and response.status_code != 200:
                print(f"   ✅ {test_case['name']}: Correctly rejected (status: {response.status_code})")
            elif not test_case["should_fail"] and response.status_code == 200:
                print(f"   ✅ {test_case['name']}: Correctly accepted")
            else:
                print(f"   ❌ {test_case['name']}: Unexpected result (status: {response.status_code})")

        except Exception as e:
            print(f"   ❌ {test_case['name']} failed: {e}")

    print("\n🎉 API testing complete!")
    return True

def wait_for_server(base_url="http://localhost:8000", max_attempts=10):
    """Wait for the server to be ready."""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is ready after {attempt + 1} attempts")
                return True
        except:
            pass

        print(f"⏳ Waiting for server... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(2)

    print("❌ Server did not become ready in time")
    return False

if __name__ == "__main__":
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"🚀 Starting API tests for {base_url}")

    # Wait for server to be ready
    if wait_for_server(base_url):
        test_api(base_url)
    else:
        print("❌ Could not connect to server. Make sure it's running!")
        sys.exit(1)