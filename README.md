# MLOps Workshop: Local to cloud

## 🎯 Workshop Overview

This hands-on workshop takes you through some of the classic tooling used for ML deployment strategies. You'll start with a "proper" traditional approach using FastAPI, Docker, and Kubernetes, then experience the dramatic simplification possible with modern serverless platforms like Modal.

**Duration:** 2-3 hours
**Audience:** ML practitioners with a trained model
**Goal:** Deploy models using both traditional and modern approaches, understand trade-offs

## 🏗️ Workshop Structure

### 📋 Part 0: Prerequisites & Setup (15 min)
**Goal:** Set up your development environment and create a sample ML model

- Install Docker, kubectl, k3d, Python 3.10+
- Generate a trained Iris classification model
- Verify all tools are working

[**→ Start Part 0**](./part-0-prerequisites/)

### 🐳 Part 1: FastAPI + Docker (40 min)
**Goal:** Create and containerize a REST API for your ML model

- **Step 1:** Run FastAPI locally (test with curl commands)
- **Step 2:** Containerize with Docker
- Learn API design, validation, error handling, containerization best practices

[**→ Start Part 1**](./part-1-docker/)

### ☸️ Part 2: Kubernetes Deployment (40 min)
**Goal:** Deploy your container to a local Kubernetes cluster

- Create k3d cluster and deploy with kubectl
- Understand Pods, Deployments, Services, LoadBalancers
- Experience the complexity of YAML configuration and orchestration

[**→ Start Part 2**](./part-2-kubernetes/)

### ☁️ (Optional Part 3): Cloud Deployment (30 min)
**Goal:** Deploy to managed Kubernetes in the cloud

- **For fast finishers:** GKE, EKS, or AKS deployment
- Container registries, production considerations
- Real-world scaling and monitoring

[**→ Start Part 3**](./part-3-cloud/)

### 🚀 Part 4: The Modal Alternative (15 min)
**Goal:** Replace all previous complexity with ~20 lines of Python

- **The Big Reveal:** Serverless deployment with Modal
- Compare approaches and understand trade-offs
- Experience the future of ML deployment

[**→ Start Part 4**](./part-4-modal/)

## 🎓 Learning Objectives

By the end of this workshop, you will:

- ✅ **Understand traditional ML deployment** - FastAPI, Docker, Kubernetes
- ✅ **Experience infrastructure complexity** - YAML files, networking, scaling
- ✅ **Appreciate modern alternatives** - Serverless, auto-scaling platforms
- ✅ **Make informed decisions** - When to use each approach
- ✅ **Deploy production-ready models** - Both traditional and modern ways


## 📊 Comparison

| Aspect | Traditional (K8s) | Modern (Modal) |
|--------|------------------|----------------|
| **Setup Time** | 60+ minutes | 2 minutes |
| **Code Lines** | ~200 lines, 8+ files | ~20 lines, 1 file |
| **Infrastructure** | Manual configuration | Fully automated |
| **Scaling** | Manual setup | Automatic (including to zero) |
| **Monitoring** | Setup required | Built-in dashboard |
| **Cost** | Always-on cluster | Pay per request |
| **SSL/HTTPS** | Manual certificates | Automatic |
| **Global Deployment** | Complex setup | One command |

## 🎯 Key Takeaways

### When to Use Traditional Kubernetes
- 🏢 Enterprise compliance requirements
- 🔧 Need full infrastructure control
- 💰 Predictable, high-volume traffic
- 🏗️ Existing Kubernetes expertise

### When to Use Modern Serverless
- 🚀 Fast iteration and prototyping
- 📊 Variable or unpredictable load
- 💡 Focus on models, not infrastructure
- 👥 Small teams without DevOps expertise

## 🚀 Quick Start

```bash
# Clone and navigate
git clone <this-repo>
cd intro-to-mlops

# Start with Part 0
cd part-0-prerequisites
python3 create_sample_model.py

# Follow the journey through each part...
```

## 🤝 Contributing

Found an issue or want to improve the workshop?
- Open issues for bugs or suggestions
- Submit PRs for improvements
- Share your experience and feedback

## 📄 License

[License](LICENSE)
