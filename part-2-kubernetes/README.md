# Part 2: Local Kubernetes Deployment

**Time Required:** 40 minutes
**Goal:** Deploy your containerized ML model to a local Kubernetes cluster using k3d

## What You'll Learn
- Kubernetes fundamentals: Pods, Deployments, Services
- How to create and apply Kubernetes manifests
- Local cluster management with k3d
- Service discovery and load balancing
- Kubernetes troubleshooting basics

## Prerequisites
- Completed Part 1 (Docker image built)
- k3d installed and working
- kubectl installed

## Step 1: Create Local Kubernetes Cluster (10 minutes)

### 1.1 Create k3d Cluster
```bash
# Create a local Kubernetes cluster
k3d cluster create mlops-demo --port "8080:80@loadbalancer"

# Verify cluster is running
kubectl cluster-info
kubectl get nodes
```

Expected output:
```
NAME                     STATUS   ROLES                  AGE   VERSION
k3d-mlops-demo-server-0   Ready    control-plane,master   30s   v1.27.4+k3s1
```

### 1.2 Load Docker Image into k3d
```bash
# Import your Docker image into the k3d cluster
k3d image import mnist-model-api:latest -c mlops-demo

# Verify the image is available
kubectl describe nodes | grep -A 5 "Non-terminated Pods"
```

## Step 2: Deploy to Kubernetes (20 minutes)

### 2.1 Apply Kubernetes Manifests
```bash
# Apply the deployment and service
kubectl apply -f deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services
```

### 2.2 Wait for Deployment
```bash
# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=mnist-model --timeout=300s

# Check pod logs
kubectl logs -l app=mnist-model
```

### 2.3 Port Forward to Access Service
```bash
# Forward local port to service
kubectl port-forward svc/mnist-model-service 8080:80 &

# Test the service
curl http://localhost:8080/health
```

## Step 3: Test the Kubernetes Deployment (10 minutes)

### 3.1 Run Health Check
```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```

### 3.2 Test Predictions
```bash
# Get sample data from the API
curl -s http://localhost:8080/sample-data | python3 -m json.tool

# Test with a sample digit (this is a 28x28=784 pixel array)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json | python3 -m json.tool

# Create a simple test with zeros (should predict digit 0 or be uncertain)
python3 -c "
import json
zeros_image = [0.0] * 784  # 28x28 black image
data = {'image': zeros_image}
with open('zeros_test.json', 'w') as f:
    json.dump(data, f)
"

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @zeros_test.json | python3 -m json.tool
```

### 3.3 Use Test Script
```bash
./test_k8s.sh
```

## Understanding Kubernetes Components

### Deployment (`deployment.yaml`)
- **Replicas**: Number of pod instances (1 initially, can be scaled)
- **Container Spec**: Docker image, ports, resource limits
- **Labels**: Used for service discovery and management
- **Health Checks**: Liveness and readiness probes

### Service (`deployment.yaml` - Service section)
- **Type**: LoadBalancer for external access
- **Selector**: Matches pods by labels
- **Ports**: Maps external port to container port
- **Load Balancing**: Distributes traffic across healthy pods

### Key Concepts
- **Pod**: Smallest deployable unit (contains our container)
- **Deployment**: Manages pod replicas and updates
- **Service**: Provides stable networking for pods
- **LoadBalancer**: Exposes service externally

## Troubleshooting

### Common Issues
1. **Pod not starting**: Check `kubectl describe pod <pod-name>`
2. **Image pull errors**: Verify image was imported with `k3d image import`
3. **Port conflicts**: Use different port in port-forward command
4. **Service not accessible**: Check service endpoints with `kubectl get endpoints`

### Useful Commands
```bash
# Check cluster status
kubectl get all

# Describe deployment
kubectl describe deployment iris-model

# View pod logs
kubectl logs -l app=iris-model -f

# Get detailed pod information
kubectl describe pods -l app=iris-model

# Check service endpoints
kubectl get endpoints iris-model-service

# Scale deployment
kubectl scale deployment iris-model --replicas=3

# Delete resources
kubectl delete -f deployment.yaml
```

## Scaling Demo (Optional)

### Scale Up
```bash
# Scale to 3 replicas
kubectl scale deployment mnist-model --replicas=3

# Watch pods come online
kubectl get pods -w

# Test load balancing
for i in {1..10}; do curl -s http://localhost:8080/health | grep -o '"status":"[^"]*"'; done
```

### Scale Down
```bash
# Scale back to 1 replica
kubectl scale deployment mnist-model --replicas=1
```

## Cleanup

```bash
# Stop port forwarding (Ctrl+C if running in foreground)
# Or kill the background process

# Delete Kubernetes resources
kubectl delete -f deployment.yaml

# Delete k3d cluster
k3d cluster delete mlops-demo
```

## Key Takeaways

- **Kubernetes provides**: Container orchestration, service discovery, load balancing, health checks
- **Complexity**: Multiple YAML files, networking concepts, debugging challenges
- **Benefits**: Scalability, reliability, production-ready infrastructure
- **Trade-offs**: Learning curve, operational overhead for simple applications

## Next Steps
Your ML model is now running on Kubernetes! In Part 3 (optional), we'll deploy to cloud Kubernetes services. In Part 4, we'll see how Modal eliminates all this complexity.

## Architecture Overview
```
[Client] → [k3d LoadBalancer] → [Service] → [Deployment] → [Pods] → [Containers]
```

Each layer adds functionality but also complexity - this is what modern platforms like Modal abstract away!
