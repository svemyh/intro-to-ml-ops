# Part 3: Cloud Deployment (Optional)

**Time Required:** 30 minutes
**Goal:** Deploy your ML model to managed Kubernetes services in the cloud
**Audience:** Students who finished Parts 1-2 quickly and want to try cloud deployment

## What You'll Learn
- Managed Kubernetes services (GKE, EKS, AKS)
- Cloud container registries
- Production deployment considerations
- Cloud-specific networking and security

## Prerequisites
- Completed Parts 1 and 2
- Cloud account (Google Cloud, AWS, or Azure)
- Cloud CLI tools installed

## Choose Your Cloud Provider

### Option A: Google Cloud Platform (GKE)
### Option B: Amazon Web Services (EKS)
### Option C: Microsoft Azure (AKS)

---

## Option A: Google Cloud Platform (GKE)

### Prerequisites
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 1. Create GKE Cluster
```bash
# Create a GKE cluster
gcloud container clusters create mlops-demo \
  --zone=us-central1-a \
  --num-nodes=2 \
  --machine-type=e2-medium \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=3

# Get cluster credentials
gcloud container clusters get-credentials mlops-demo --zone=us-central1-a
```

### 2. Push Image to Container Registry
```bash
# Tag the image for Google Container Registry
docker tag iris-model-api:latest gcr.io/YOUR_PROJECT_ID/iris-model-api:latest

# Push to registry
docker push gcr.io/YOUR_PROJECT_ID/iris-model-api:latest
```

### 3. Update Deployment for GKE
```bash
# Update the image in deployment-gke.yaml
sed -i 's/iris-model-api:latest/gcr.io\/YOUR_PROJECT_ID\/iris-model-api:latest/' deployment-gke.yaml

# Apply deployment
kubectl apply -f deployment-gke.yaml

# Get external IP
kubectl get services iris-model-service -w
```

### 4. Test Cloud Deployment
```bash
# Get external IP
EXTERNAL_IP=$(kubectl get service iris-model-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test the API
curl http://$EXTERNAL_IP/health
curl -X POST http://$EXTERNAL_IP/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### 5. Cleanup GKE
```bash
# Delete the cluster
gcloud container clusters delete mlops-demo --zone=us-central1-a
```

---

## Option B: Amazon Web Services (EKS)

### Prerequisites
```bash
# Install AWS CLI and eksctl
# https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html
# https://docs.aws.amazon.com/eks/latest/userguide/eksctl.html

# Configure AWS credentials
aws configure
```

### 1. Create EKS Cluster
```bash
# Create EKS cluster (takes 10-15 minutes)
eksctl create cluster \
  --name mlops-demo \
  --version 1.27 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed
```

### 2. Push Image to ECR
```bash
# Create ECR repository
aws ecr create-repository --repository-name iris-model-api --region us-west-2

# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

# Tag and push image
docker tag iris-model-api:latest ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/iris-model-api:latest
docker push ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/iris-model-api:latest
```

### 3. Deploy to EKS
```bash
# Update deployment-eks.yaml with your ECR image URI
# Apply deployment
kubectl apply -f deployment-eks.yaml

# Get load balancer URL
kubectl get services iris-model-service
```

### 4. Cleanup EKS
```bash
# Delete cluster
eksctl delete cluster --name mlops-demo --region us-west-2
```

---

## Option C: Microsoft Azure (AKS)

### Prerequisites
```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login to Azure
az login

# Create resource group
az group create --name mlops-demo-rg --location eastus
```

### 1. Create AKS Cluster
```bash
# Create AKS cluster
az aks create \
  --resource-group mlops-demo-rg \
  --name mlops-demo \
  --node-count 2 \
  --node-vm-size Standard_B2s \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group mlops-demo-rg --name mlops-demo
```

### 2. Push Image to ACR
```bash
# Create Azure Container Registry
az acr create --resource-group mlops-demo-rg --name mlopsdemoacr --sku Basic --admin-enabled true

# Login to ACR
az acr login --name mlopsdemoacr

# Tag and push image
docker tag iris-model-api:latest mlopsdemoacr.azurecr.io/iris-model-api:latest
docker push mlopsdemoacr.azurecr.io/iris-model-api:latest
```

### 3. Deploy to AKS
```bash
# Update deployment-aks.yaml with your ACR image
# Apply deployment
kubectl apply -f deployment-aks.yaml

# Get external IP
kubectl get services iris-model-service
```

### 4. Cleanup AKS
```bash
# Delete resource group (deletes everything)
az group delete --name mlops-demo-rg --yes --no-wait
```

---

## Production Considerations

### Security
- Use private container registries
- Enable RBAC (Role-Based Access Control)
- Network policies for pod-to-pod communication
- Secrets management for sensitive data

### Monitoring
- Container insights and logging
- Application performance monitoring
- Health checks and alerting
- Resource usage monitoring

### Scalability
- Horizontal Pod Autoscaler (HPA)
- Vertical Pod Autoscaler (VPA)
- Cluster autoscaling
- Multi-zone deployments

### CI/CD Integration
- Automated builds and deployments
- GitOps workflows
- Blue-green or rolling deployments
- Automated testing in pipelines

## Cost Optimization

### Tips to Save Money
- Use spot/preemptible instances for development
- Set up cluster autoscaling
- Use smaller node types when possible
- Delete clusters when not in use
- Monitor resource usage

### Estimated Costs (USD/month)
- **GKE**: $50-150 for small dev cluster
- **EKS**: $70-200 (includes EKS control plane fee)
- **AKS**: $50-150 for small dev cluster

## Key Takeaways

### What You've Learned
- **Managed Kubernetes**: Cloud providers handle control plane
- **Container Registries**: Secure, scalable image storage
- **Cloud Integration**: Networking, security, monitoring built-in
- **Production Complexity**: Many more considerations than local development

### Trade-offs
- **Pros**: Production-ready, scalable, managed infrastructure
- **Cons**: Complex setup, ongoing costs, cloud lock-in
- **Alternative**: Modern platforms (Part 4) eliminate much of this complexity

## Next Steps
In Part 4, you'll see how Modal makes all of this complexity disappear with just a few lines of Python code!