# Part 2: Kubernetes Deployment - Visual Diagrams

This document contains Mermaid diagrams illustrating key concepts from Part 2 of the MLOps workshop.

## 1. Kubernetes Architecture Overview

```mermaid
graph TB
    subgraph "k3d Local Cluster"
        subgraph "Control Plane"
            A[API Server] --> B[etcd<br/>Cluster State]
            A --> C[Scheduler]
            A --> D[Controller Manager]
        end

        subgraph "Worker Node"
            E[kubelet] --> F[Container Runtime<br/>Docker/containerd]
            G[kube-proxy] --> H[Network Rules]
        end

        subgraph "Our Application"
            I[Deployment] --> J[ReplicaSet]
            J --> K[Pod 1<br/>iris-model container]
            L[Service] --> K
            M[LoadBalancer] --> L
        end
    end

    subgraph "External Access"
        N[Client<br/>kubectl/curl] --> A
        O[Port Forward<br/>8080:80] --> M
    end

    style K fill:#c8e6c9
    style L fill:#e3f2fd
    style M fill:#fff3e0
```

## 2. Pod Lifecycle and Management

```mermaid
stateDiagram-v2
    [*] --> Pending: kubectl apply
    Pending --> ContainerCreating: Scheduled to Node
    ContainerCreating --> Running: Image Pulled & Started

    Running --> Ready: Health Checks Pass
    Ready --> Serving: Receiving Traffic

    Serving --> Terminating: kubectl delete / Scale Down
    Terminating --> [*]: Graceful Shutdown

    ContainerCreating --> Failed: Image Pull Error
    Running --> CrashLoopBackOff: Application Error
    CrashLoopBackOff --> Running: Restart Policy
    Failed --> [*]: Max Retries Exceeded

    note right of Ready: readinessProbe\n/health endpoint
    note right of Serving: livenessProbe\nmonitoring
```

## 3. Deployment Resource Hierarchy

```mermaid
graph TD
    A[Deployment<br/>iris-model] --> B[ReplicaSet<br/>iris-model-abc123]
    B --> C[Pod<br/>iris-model-abc123-xyz89]
    C --> D[Container<br/>iris-model:latest]

    subgraph "Deployment Spec"
        E[replicas: 1]
        F[selector: app=iris-model]
        G[template: Pod spec]
    end

    subgraph "Pod Spec"
        H[containers: iris-model]
        I[resources: limits/requests]
        J[probes: health checks]
        K[ports: 80]
    end

    A -.-> E
    A -.-> F
    A -.-> G
    C -.-> H
    C -.-> I
    C -.-> J
    C -.-> K

    style A fill:#e3f2fd
    style C fill:#c8e6c9
    style D fill:#fff3e0
```

## 4. Service Discovery and Networking

```mermaid
graph LR
    subgraph "Client"
        A[kubectl port-forward<br/>localhost:8080]
    end

    subgraph "k3d Cluster Network"
        B[Service<br/>iris-model-service<br/>ClusterIP: 10.43.x.x]

        subgraph "Endpoints"
            C[Pod 1<br/>10.42.0.x:80]
            D[Pod 2<br/>10.42.0.y:80<br/>(if scaled)]
        end

        subgraph "Service Types"
            E[LoadBalancer<br/>External Access]
            F[ClusterIP<br/>Internal Only]
            G[NodePort<br/>Node IP + Port]
        end
    end

    A --> B
    B --> C
    B --> D
    B -.-> E
    B -.-> F
    B -.-> G

    style B fill:#e3f2fd
    style C fill:#c8e6c9
    style D fill:#c8e6c9
```

## 5. kubectl Command Flow

```mermaid
sequenceDiagram
    participant User
    participant kubectl
    participant APIServer
    participant etcd
    participant Scheduler
    participant kubelet
    participant Container

    User->>kubectl: kubectl apply -f deployment.yaml
    kubectl->>APIServer: Create Deployment
    APIServer->>etcd: Store Deployment Spec
    etcd-->>APIServer: Stored

    APIServer->>Scheduler: Schedule Pod
    Scheduler->>APIServer: Assign to Node
    APIServer->>kubelet: Create Pod
    kubelet->>Container: Pull & Start Image
    Container-->>kubelet: Running
    kubelet->>APIServer: Pod Status Update
    APIServer->>User: Deployment Created
```

## 6. Health Check Configuration

```mermaid
graph TB
    subgraph "Container Health Checks"
        A[Container Starts] --> B{Readiness Probe}
        B -->|Pass| C[Pod Ready for Traffic]
        B -->|Fail| D[Pod Not in Service]
        D --> E[Wait & Retry]
        E --> B

        C --> F{Liveness Probe}
        F -->|Pass| G[Container Healthy]
        F -->|Fail| H[Restart Container]
        H --> A

        subgraph "Probe Configuration"
            I[httpGet: /health]
            J[port: 80]
            K[initialDelaySeconds: 30]
            L[periodSeconds: 10]
            M[timeoutSeconds: 5]
            N[failureThreshold: 3]
        end
    end

    style C fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#ffcdd2
    style D fill:#fff3e0
```

## 7. Resource Management

```mermaid
graph TD
    subgraph "Pod Resource Specification"
        A[resources:]
        A --> B[requests:]
        A --> C[limits:]

        B --> D[memory: 256Mi<br/>Guaranteed Allocation]
        B --> E[cpu: 250m<br/>0.25 CPU cores]

        C --> F[memory: 512Mi<br/>Maximum Usage]
        C --> G[cpu: 500m<br/>0.5 CPU cores]
    end

    subgraph "Quality of Service Classes"
        H[Guaranteed<br/>requests = limits]
        I[Burstable<br/>requests < limits]
        J[BestEffort<br/>no resources set]
    end

    subgraph "Node Resource Management"
        K[Node Allocatable Resources] --> L{Enough Resources?}
        L -->|Yes| M[Schedule Pod]
        L -->|No| N[Pod Pending]
    end

    D --> H
    F --> I
    M --> O[Pod Running]
    N --> P[Wait for Resources]

    style O fill:#c8e6c9
    style N fill:#fff3e0
    style P fill:#ffcdd2
```

## 8. Scaling and Load Balancing

```mermaid
graph TB
    A[Initial State<br/>1 Replica] --> B[Scale Command<br/>kubectl scale deployment iris-model --replicas=3]

    B --> C{Deployment Controller}
    C --> D[Create 2 New Pods]
    D --> E[Pod 2: Pending]
    D --> F[Pod 3: Pending]

    E --> G[Pod 2: Running]
    F --> H[Pod 3: Running]

    subgraph "Load Balancing"
        I[Service<br/>iris-model-service] --> J[Pod 1<br/>Ready]
        I --> K[Pod 2<br/>Ready]
        I --> L[Pod 3<br/>Ready]

        M[Client Request] --> I
        I -.->|Round Robin| J
        I -.->|Round Robin| K
        I -.->|Round Robin| L
    end

    G --> K
    H --> L

    style I fill:#e3f2fd
    style J fill:#c8e6c9
    style K fill:#c8e6c9
    style L fill:#c8e6c9
```

## 9. YAML Manifest Structure

```mermaid
graph LR
    subgraph "deployment.yaml"
        A[Deployment Manifest]
        A --> B[apiVersion: apps/v1<br/>kind: Deployment]
        A --> C[metadata:<br/>name, labels]
        A --> D[spec:<br/>replicas, selector, template]

        D --> E[template.spec:<br/>containers definition]
        E --> F[container:<br/>name, image, ports]
        E --> G[resources:<br/>requests, limits]
        E --> H[probes:<br/>liveness, readiness]

        A2[Service Manifest]
        A2 --> I[apiVersion: v1<br/>kind: Service]
        A2 --> J[metadata:<br/>name, labels]
        A2 --> K[spec:<br/>type, selector, ports]

        K --> L[type: LoadBalancer]
        K --> M[selector: app=iris-model]
        K --> N[ports: 80 -> 80]
    end

    style A fill:#e3f2fd
    style A2 fill:#fff3e0
    style F fill:#c8e6c9
```

## 10. Troubleshooting Decision Tree

```mermaid
flowchart TD
    A[Application Not Working] --> B{Can you access the service?}
    B -->|No| C{Is port-forward running?}
    B -->|Yes| D{Getting correct response?}

    C -->|No| E[Run: kubectl port-forward svc/iris-model-service 8080:80]
    C -->|Yes| F{Is service configured correctly?}

    F -->|Check| G[kubectl get svc iris-model-service]
    G --> H{Endpoints exist?}
    H -->|No| I[kubectl get endpoints iris-model-service]
    H -->|Yes| J[Check pod status]

    I --> K{Pods running?}
    K -->|No| L[kubectl get pods -l app=iris-model]
    K -->|Yes| M[Check selector labels match]

    L --> N{Pod status?}
    N -->|Pending| O[Check: kubectl describe pod]
    N -->|CrashLoopBackOff| P[Check: kubectl logs pod-name]
    N -->|ImagePullBackOff| Q[Check: k3d image import]

    D -->|No| R[Check application logs]
    R --> S[kubectl logs -l app=iris-model]

    style E fill:#c8e6c9
    style S fill:#c8e6c9
    style O fill:#fff3e0
    style P fill:#ffcdd2
    style Q fill:#ffcdd2
```

## 11. k3d Cluster Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Creating: k3d cluster create
    Creating --> Running: Cluster Ready
    Running --> ImageImport: k3d image import
    ImageImport --> Deploying: kubectl apply
    Deploying --> ApplicationRunning: Pods Ready

    ApplicationRunning --> Scaling: kubectl scale
    Scaling --> ApplicationRunning: Replicas Updated

    ApplicationRunning --> PortForwarding: kubectl port-forward
    PortForwarding --> Testing: curl/test scripts
    Testing --> ApplicationRunning: Tests Complete

    ApplicationRunning --> Cleanup: kubectl delete
    Cleanup --> Running: Resources Removed
    Running --> Deleting: k3d cluster delete
    Deleting --> [*]: Cluster Destroyed

    Creating --> Failed: Creation Error
    Failed --> [*]: Fix Issues & Retry
```

## 12. Container to Kubernetes Migration

```mermaid
graph LR
    subgraph "Part 1: Docker"
        A[Docker Image<br/>iris-model-api:latest]
        B[docker run -p 8080:80<br/>iris-model-api]
        C[Single Container<br/>Manual Management]
    end

    subgraph "Part 2: Kubernetes"
        D[Same Docker Image<br/>iris-model-api:latest]
        E[k3d image import<br/>Load into cluster]
        F[kubectl apply<br/>deployment.yaml]
        G[Pod Orchestration<br/>Automatic Management]

        subgraph "Kubernetes Benefits"
            H[Health Checks<br/>Auto Restart]
            I[Service Discovery<br/>Load Balancing]
            J[Scaling<br/>Multiple Replicas]
            K[Resource Management<br/>CPU/Memory Limits]
        end
    end

    A --> D
    B -.-> F
    C -.-> G
    G --> H
    G --> I
    G --> J
    G --> K

    style A fill:#fff3e0
    style D fill:#fff3e0
    style G fill:#c8e6c9
    style H fill:#e3f2fd
    style I fill:#e3f2fd
    style J fill:#e3f2fd
    style K fill:#e3f2fd
```

## 13. Network Traffic Flow

```mermaid
graph TD
    subgraph "External"
        A[Developer<br/>localhost:8080]
    end

    subgraph "k3d Host Network"
        B[Port Forward<br/>kubectl proxy]
    end

    subgraph "k3d Cluster Network"
        C[LoadBalancer Service<br/>iris-model-service]

        subgraph "Pod Network"
            D[Pod IP: 10.42.0.5<br/>Port: 80]
            E[iris-model container<br/>uvicorn server]
            F[FastAPI Application<br/>/health, /predict]
        end
    end

    A -->|HTTP Request| B
    B -->|Forward to Cluster| C
    C -->|Route to Pod| D
    D -->|Container Port| E
    E -->|ASGI| F
    F -->|Response| E
    E -->|Response| D
    D -->|Response| C
    C -->|Response| B
    B -->|Response| A

    style F fill:#c8e6c9
    style C fill:#e3f2fd
    style D fill:#fff3e0
```

## 14. Kubernetes vs Docker Comparison

```mermaid
graph TB
    subgraph "Docker Approach (Part 1)"
        A1[Manual Container Management]
        A1 --> A2[docker run commands]
        A1 --> A3[Manual port mapping]
        A1 --> A4[No automatic restart]
        A1 --> A5[Single container instance]
        A1 --> A6[Manual health monitoring]
    end

    subgraph "Kubernetes Approach (Part 2)"
        B1[Orchestrated Container Management]
        B1 --> B2[kubectl apply declarative]
        B1 --> B3[Service-based networking]
        B1 --> B4[Automatic restart on failure]
        B1 --> B5[Scalable replica management]
        B1 --> B6[Built-in health checks]
        B1 --> B7[Resource management]
        B1 --> B8[Rolling updates]
    end

    style A1 fill:#ffcdd2
    style B1 fill:#c8e6c9
    style B7 fill:#e3f2fd
    style B8 fill:#e3f2fd
```

## 15. Complete Deployment Workflow

```mermaid
flowchart TD
    A[Start: Docker image ready] --> B[Create k3d cluster]
    B --> C[Import Docker image to cluster]
    C --> D[Apply deployment.yaml]
    D --> E[Wait for pods to be ready]

    E --> F{Pods healthy?}
    F -->|No| G[Check logs & debug]
    G --> H[Fix issues]
    H --> D

    F -->|Yes| I[Start port-forward]
    I --> J[Test health endpoint]
    J --> K[Test prediction endpoint]
    K --> L[Run comprehensive tests]

    L --> M{All tests pass?}
    M -->|No| G
    M -->|Yes| N[✅ Deployment successful]

    N --> O[Optional: Scale deployment]
    O --> P[Monitor and maintain]

    P --> Q[When done: Clean up]
    Q --> R[kubectl delete resources]
    R --> S[k3d cluster delete]

    style N fill:#c8e6c9
    style B fill:#e3f2fd
    style G fill:#fff3e0
    style S fill:#ffcdd2
```

---

## How to Use These Diagrams

1. **Copy the Mermaid code** from any diagram above
2. **Paste into your preferred tool:**
   - GitHub/GitLab (native support)
   - Mermaid Live Editor (https://mermaid.live/)
   - VS Code with Mermaid extension
   - Notion, Obsidian, or other markdown tools

3. **Use for teaching:**
   - Show before hands-on exercises
   - Reference during troubleshooting
   - Explain complex Kubernetes concepts visually

These diagrams complement the hands-on Kubernetes experience in Part 2, helping students understand the orchestration concepts they're implementing!
