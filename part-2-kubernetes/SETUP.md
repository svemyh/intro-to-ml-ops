# Installation guide - `k3d` & `kubectl`

- https://k3d.io/#installation
- https://kubernetes.io/docs/tasks/tools/#kubectl

## Linux

### k3d
```bash
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
```

### kubectl
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## macOS (not tested)

### k3d
```bash
# Using Homebrew
brew install k3d

# Or using curl
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
```

### kubectl
```bash
# Using Homebrew
brew install kubectl

# Or using curl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
```

## Windows (not tested)

### k3d
```powershell
# Download from GitHub releases
# https://github.com/k3d-io/k3d/releases

# Or install using Chocolatey
choco install k3d

# or using Scoop
scoop install k3d

```

### kubectl
```powershell
# Using Chocolatey
choco install kubernetes-cli

# Using Scoop
scoop install kubectl

# Or download from Kubernetes releases
# https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/
```

## Verify Installation

```bash
k3d version
kubectl version --client
```
