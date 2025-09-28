# MLOps Workshop: From Kubernetes to Modal

> **The Journey from Complex to Simple**
> Experience the full spectrum of ML deployment - from traditional infrastructure to modern serverless platforms

## 🎯 Workshop Overview

This hands-on workshop takes you through the evolution of ML deployment strategies. You'll start with the "proper" traditional approach using FastAPI, Docker, and Kubernetes, then experience the dramatic simplification possible with modern serverless platforms like Modal.

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

### ☁️ Part 3: Cloud Deployment (30 min - Optional)
**Goal:** Deploy to managed Kubernetes in the cloud

- **For fast finishers:** GKE, EKS, or AKS deployment
- Container registries, production considerations
- Real-world scaling and monitoring

[**→ Start Part 3**](./part-3-cloud/)

### 🚀 Part 4: The Modal Alternative (20 min)
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

## 🛠️ What You'll Build

### Traditional Stack (Parts 1-3)
```
[FastAPI App] → [Docker Container] → [Kubernetes Pod] → [Cloud LoadBalancer]
```

**Files Created:**
- `app.py` - FastAPI application with model serving
- `Dockerfile` - Container configuration
- `deployment.yaml` - Kubernetes manifests
- Multiple configuration and test files

**Complexity:** ~200 lines across 8+ files

### Modern Stack (Part 4)
```
[Modal Function] → [Auto-scaling Infrastructure] → [Global HTTPS Endpoints]
```

**Files Created:**
- `app_modal.py` - Complete serverless application

**Complexity:** ~20 lines in 1 file

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

## 📁 Repository Structure

```
intro-to-mlops/
├── README.md                 # This file
├── plan.md                   # Implementation plan
├── part-0-prerequisites/     # Environment setup + model creation
├── part-1-docker/           # FastAPI + Docker
├── part-2-kubernetes/       # K8s deployment
├── part-3-cloud/           # Cloud deployment (optional)
└── part-4-modal/           # Serverless alternative
```

## 🎉 Success Stories

> *"I went from spending 60% of my time on infrastructure to 100% on ML models"*
> — Workshop Participant

> *"The Modal deployment literally took 30 seconds vs. 2 hours for Kubernetes"*
> — ML Engineer

> *"Finally understood when to use each approach - game changer for our team decisions"*
> — Tech Lead

## 🤝 Contributing

Found an issue or want to improve the workshop?
- Open issues for bugs or suggestions
- Submit PRs for improvements
- Share your experience and feedback

## 📄 License

This workshop is open source and free to use for educational purposes.

---

**Ready to start your MLOps journey?**

👉 **[Begin with Part 0: Prerequisites](./part-0-prerequisites/)**

*Experience the evolution from complex to simple, and make informed decisions about your ML deployment strategy.*