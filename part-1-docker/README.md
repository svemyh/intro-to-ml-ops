# Part 1: FastAPI + Docker Deployment

**Goal:** Create a FastAPI wrapper for your ML model, test it locally, then containerize it

## What You'll Learn
- How to create a REST API around an ML model using FastAPI
- How to containerize Python applications with Docker

## Step 1: Run FastAPI App Locally

### 1.1 Copy Your Trained Model
Copy the model files from Part 0:
```bash
cp ../part-0-prerequisites/mnist_model.pth .
cp ../part-0-prerequisites/sample_data.json .
```

### 1.2 Install Dependencies
```bash
uv sync # (execute in the directory notebooks/town_hall_02)
```

### 1.3 Run the FastAPI Application
```bash
python3 app.py
```

You should see somethign like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 1.4 Test the API
Open the client interface in your browser client/index.html


## Step 2: Containerize with Docker

### 2.1 Build Docker Image
```bash
docker build -t mnist-model-api .
```

### 2.2 Run Container
```bash
docker run -p 8080:80 mnist-model-api
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


## Troubleshooting

### Common Issues
0. **Install**: remember to install and start Docker desktop if on Mac or Windows
1. **Port already in use**: Try a different port with `-p 8081:80`
2. **Model not found**: Make sure you copied the model file
3. **Permission denied**: On Linux, you might need `sudo docker`


## Next Steps
Your application is now containerized! In Part 2, we'll deploy this container to Kubernetes.

## Key Takeaways
- FastAPI makes it easy to create APIs around ML models
- Docker provides consistent deployment environments
- Container orchestration (coming next) enables scaling and management
