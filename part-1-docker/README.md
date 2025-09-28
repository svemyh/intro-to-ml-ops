# Part 1: FastAPI + Docker Deployment

**Time Required:** 40 minutes
**Goal:** Create a FastAPI wrapper for your ML model, test it locally, then containerize it

## What You'll Learn
- How to create a REST API around an ML model using FastAPI
- How to containerize Python applications with Docker
- Testing HTTP endpoints with curl
- Docker best practices for ML applications

## Step 1: Run FastAPI App Locally (15 minutes)

### 1.1 Copy Your Trained Model
Copy the model files from Part 0:
```bash
cp ../part-0-prerequisites/iris_model.pkl .
cp ../part-0-prerequisites/sample_data.json .
```

### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Run the FastAPI Application
```bash
python3 app.py
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 1.4 Test the API
Open a new terminal and test the endpoint:

```bash
# Test health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Test with different samples
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.2, 2.8, 4.8, 1.8]}'
```

Expected response:
```json
{"prediction": 0, "class_name": "setosa", "confidence": 0.95}
```

### 1.5 View API Documentation
Visit http://localhost:8000/docs to see the automatically generated API documentation.

## Step 2: Containerize with Docker (25 minutes)

### 2.1 Build Docker Image
```bash
docker build -t iris-model-api .
```

### 2.2 Run Container
```bash
docker run -p 8080:80 iris-model-api
```

### 2.3 Test Containerized API
```bash
# Test health check
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### 2.4 Check Container Status
```bash
# List running containers
docker ps

# View container logs
docker logs <container_id>

# Stop container
docker stop <container_id>
```

## Understanding the Components

### FastAPI Application (`app.py`)
- **Model Loading**: Loads the trained model once at startup
- **Health Check**: Simple endpoint to verify the service is running
- **Prediction Endpoint**: Accepts features and returns predictions
- **Validation**: Uses Pydantic for request validation
- **Error Handling**: Graceful error responses

### Docker Configuration (`Dockerfile`)
- **Base Image**: Python 3.11 slim for smaller size
- **Dependencies**: Install requirements first (better caching)
- **Application Code**: Copy and run the application
- **Port Exposure**: Service runs on port 80 inside container

### Requirements (`requirements.txt`)
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **ML Libraries**: scikit-learn, joblib, numpy

## Troubleshooting

### Common Issues
1. **Port already in use**: Try a different port with `-p 8081:80`
2. **Model not found**: Make sure you copied the model file
3. **Permission denied**: On Linux, you might need `sudo docker`

### Verification Commands
```bash
# Check if port is in use
netstat -tlnp | grep :8000

# Check Docker is running
docker --version

# View container resource usage
docker stats
```

## Next Steps
Your application is now containerized! In Part 2, we'll deploy this container to Kubernetes.

## Key Takeaways
- FastAPI makes it easy to create APIs around ML models
- Docker provides consistent deployment environments
- Container orchestration (coming next) enables scaling and management