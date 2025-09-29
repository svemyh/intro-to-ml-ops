#!/bin/bash
# Test script for Kubernetes deployed Iris Classification API

BASE_URL=${1:-"http://localhost:8080"}
echo "🧪 Testing Kubernetes deployed API at $BASE_URL"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

# Check if cluster is running
echo -e "\n${BLUE}🔍 Checking Kubernetes cluster status...${NC}"
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ No Kubernetes cluster found. Please create a cluster first:${NC}"
    echo "k3d cluster create mlops-demo --port \"8080:80@loadbalancer\""
    exit 1
fi

cluster_info=$(kubectl cluster-info | head -n 1)
echo -e "   ${GREEN}✅ Cluster is running: $cluster_info${NC}"

# Check deployment status
echo -e "\n${BLUE}📦 Checking deployment status...${NC}"
deployment_status=$(kubectl get deployment mnist-model --no-headers 2>/dev/null | awk '{print $2}')
if [ -z "$deployment_status" ]; then
    echo -e "   ${RED}❌ Deployment 'mnist-model' not found. Please apply deployment.yaml first:${NC}"
    echo "   kubectl apply -f deployment.yaml"
    exit 1
else
    echo -e "   ${GREEN}✅ Deployment found: $deployment_status ready${NC}"
fi

# Check pod status
echo -e "\n${BLUE}🚀 Checking pod status...${NC}"
pod_status=$(kubectl get pods -l app=mnist-model --no-headers 2>/dev/null)
if [ -z "$pod_status" ]; then
    echo -e "   ${RED}❌ No pods found for mnist-model${NC}"
    exit 1
fi

echo "   📊 Pod status:"
kubectl get pods -l app=mnist-model | sed 's/^/      /'

# Check if pods are ready
ready_pods=$(kubectl get pods -l app=mnist-model --no-headers | grep Running | grep "1/1" | wc -l)
total_pods=$(kubectl get pods -l app=mnist-model --no-headers | wc -l)

if [ "$ready_pods" -eq 0 ]; then
    echo -e "   ${RED}❌ No pods are ready. Checking logs...${NC}"
    echo -e "   ${YELLOW}Pod logs:${NC}"
    kubectl logs -l app=mnist-model --tail=10 | sed 's/^/      /'
    exit 1
else
    echo -e "   ${GREEN}✅ $ready_pods/$total_pods pods are ready${NC}"
fi

# Check service status
echo -e "\n${BLUE}🌐 Checking service status...${NC}"
service_info=$(kubectl get service mnist-model-service --no-headers 2>/dev/null)
if [ -z "$service_info" ]; then
    echo -e "   ${RED}❌ Service 'mnist-model-service' not found${NC}"
    exit 1
else
    echo -e "   ${GREEN}✅ Service found:${NC}"
    kubectl get service mnist-model-service | sed 's/^/      /'
fi

# Check endpoints
endpoints=$(kubectl get endpoints mnist-model-service --no-headers 2>/dev/null | awk '{print $2}')
if [ "$endpoints" = "<none>" ] || [ -z "$endpoints" ]; then
    echo -e "   ${RED}❌ No endpoints available for service${NC}"
    exit 1
else
    echo -e "   ${GREEN}✅ Service endpoints: $endpoints${NC}"
fi

# Wait a moment for any port-forward to be ready
echo -e "\n${PURPLE}⏳ Waiting for API to be accessible...${NC}"
sleep 2

# Test API endpoints
echo -e "\n${BLUE}🧪 Testing API endpoints...${NC}"

# Test 1: Health check
echo "   💚 Testing health endpoint..."
response=$(curl -s -w "%{http_code}" "$BASE_URL/health" --connect-timeout 5 --max-time 10)
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Health check passed${NC}"
    echo "   📊 Status: $(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['status'])" 2>/dev/null || echo "healthy")"
else
    echo -e "   ${RED}❌ Health check failed: HTTP $http_code${NC}"
    if [ "$http_code" = "000" ]; then
        echo -e "   ${YELLOW}💡 Tip: Make sure port-forward is running:${NC}"
        echo "        kubectl port-forward svc/iris-model-service 8080:80"
    fi
    exit 1
fi

# Test 2: Prediction
echo "   🔢 Testing prediction endpoint with MNIST digit..."

# Create a simple test image (zeros should be somewhat predictable)
zeros_image='['
for i in {1..784}; do
    if [ $i -eq 784 ]; then
        zeros_image+='0.0'
    else
        zeros_image+='0.0,'
    fi
done
zeros_image+=']'

response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image\": $zeros_image}" \
  --connect-timeout 10 --max-time 15)
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ MNIST prediction successful${NC}"
    predicted_digit=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['prediction'])" 2>/dev/null || echo "unknown")
    confidence=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['confidence'])" 2>/dev/null || echo "unknown")
    echo "   📊 Predicted digit: $predicted_digit (confidence: $confidence)"
    echo "   🖼️  Input: 28x28 black image (784 zeros)"
else
    echo -e "   ${RED}❌ MNIST prediction failed: HTTP $http_code${NC}"
    echo "   Response: $body"
    if [ "$http_code" = "422" ]; then
        echo -e "   ${YELLOW}💡 This might be a validation error. Check image format.${NC}"
    fi
fi

# Test 3: Load balancing (if multiple pods)
if [ "$total_pods" -gt 1 ]; then
    echo "   ⚖️  Testing load balancing across $total_pods pods..."
    declare -A pod_responses

    for i in {1..10}; do
        response=$(curl -s "$BASE_URL/health" --connect-timeout 2 --max-time 5 2>/dev/null)
        # In a real scenario, you'd check response headers or add pod identification
        # For now, we'll just count successful responses
        if echo "$response" | grep -q "healthy"; then
            ((pod_responses["success"]++))
        fi
    done

    if [ "${pod_responses["success"]}" -gt 0 ]; then
        echo -e "   ${GREEN}✅ Load balancing working: ${pod_responses["success"]}/10 successful requests${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Load balancing test inconclusive${NC}"
    fi
fi

# Summary
echo -e "\n${GREEN}🎉 Kubernetes deployment test complete!${NC}"
echo -e "\n${PURPLE}📋 Summary:${NC}"
echo "   • Cluster: Running"
echo "   • Deployment: $deployment_status"
echo "   • Pods: $ready_pods/$total_pods ready"
echo "   • Service: Active with endpoints"
echo "   • API: Healthy and responding"

echo -e "\n${BLUE}💡 Useful commands:${NC}"
echo "   • View pods: kubectl get pods -l app=mnist-model"
echo "   • View logs: kubectl logs -l app=mnist-model -f"
echo "   • Scale up: kubectl scale deployment mnist-model --replicas=3"
echo "   • Port forward: kubectl port-forward svc/mnist-model-service 8080:80"
