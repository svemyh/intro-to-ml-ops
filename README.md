# MLOps Workshop: From local to cloud

**Duration:** 2-3 hours
**Audience:** ML practitioners with a trained model
**Goal:** Deploy models using both traditional and modern approaches, understand trade-offs

## Workshop Structure

### Part 0: Prerequisites & Setup

- Install Docker, kubectl, k3d, Python 3.10+

[**→ Start Part 0**](./part-0-prerequisites/)

### Part 1: FastAPI + Docker 
**Goal:** Create and containerize a REST API for your ML model

- **Step 1:** Run FastAPI locally (test with curl commands)
- **Step 2:** Containerize with Docker
- Learn API design, validation, error handling, containerization best practices

[**→ Start Part 1**](./part-1-docker/)

### Part 2: Kubernetes Deployment 
**Goal:** Deploy your container to a local Kubernetes cluster

[**→ Start Part 2**](./part-2-kubernetes/)

### (Optional Part 3): Cloud Deployment 
**Goal:** Deploy to managed Kubernetes in the cloud

[**→ Start Part 3**](./part-3-cloud/)

### 🚀 Part 4: The Modal Alternative
**Goal:** Replace all previous complexity with ~20 lines of Python

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



## Comparison

### When to Use Traditional Kubernetes
- Enterprise compliance requirements
- Need full infrastructure control
- Predictable, high-volume traffic
- Existing Kubernetes expertise

### When to Use Modern Serverless
- Fast iteration and prototyping
- Variable or unpredictable load
- Focus on models, not infrastructure
- Small teams without DevOps expertise

## Quick Start

```bash
# Clone and navigate
git clone <this-repo>
cd intro-to-mlops

# Start with Part 0
cd part-0-prerequisites
python3 create_sample_model.py

# Follow the journey through each part...
```

## License

[License](LICENSE)

## Credits

Thanks to ReLU NTNU