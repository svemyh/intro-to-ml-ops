# Part 0: Prerequisites & Setup

**Time Required:** 15 minutes
**Goal:** Set up your development environment and create a sample ML model

## Prerequisites Installation

### For All Platforms
- Python 3.10 or higher
- Git
- Text editor (VS Code recommended)

### Platform-Specific Instructions

#### 🐧 Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for docker group to take effect

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install k3d
wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
```

#### 🍎 macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python3
brew install --cask docker
brew install kubectl
brew install k3d

# Start Docker Desktop
open /Applications/Docker.app
```

#### 🪟 Windows
1. **Install Python:**
   - Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

2. **Install Docker Desktop:**
   - Download from https://www.docker.com/products/docker-desktop
   - Enable WSL 2 integration if prompted

3. **Install kubectl:**
   ```powershell
   # Using Chocolatey (install Chocolatey first if needed)
   choco install kubernetes-cli

   # OR download directly
   # Download from https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/
   ```

4. **Install k3d:**
   ```powershell
   # Using Chocolatey
   choco install k3d

   # OR download from https://github.com/k3d-io/k3d/releases
   ```

## Verification

Run these commands to verify your installation:

```bash
python3 --version
docker --version
kubectl version --client
k3d version
```

Expected output should show version numbers for all tools.

## Next Steps

1. Run `python3 create_mnist_model.py` to generate a trained MNIST digit classification model
2. Verify the model files are created:
   - `mnist_model.pth` - Complete PyTorch model (97.5% accuracy)
   - `sample_data.json` - API testing data with digit samples
   - `inference_utils.py` - Helper functions for prediction
3. You're ready for Part 1!

## Troubleshooting

- **Docker permission issues (Linux):** Make sure you logged out and back in after adding user to docker group
- **k3d not found:** Make sure `/usr/local/bin` is in your PATH
- **Python version too old:** This workshop requires Python 3.10+
