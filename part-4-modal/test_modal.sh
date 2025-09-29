#!/bin/bash
# Test script for Modal deployed Iris Classification API

BASE_URL=${1}
if [ -z "$BASE_URL" ]; then
    echo "Usage: $0 <modal-base-url>"
    echo "Example: $0 https://username--iris-classifier"
    exit 1
fi

# Remove trailing slash
BASE_URL=${BASE_URL%/}

echo "🧪 Testing Modal deployed API at $BASE_URL"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test 1: Health check
echo -e "\n${BLUE}1. Testing health endpoint...${NC}"
health_url="${BASE_URL}-health.modal.run"
response=$(curl -s -w "%{http_code}" "$health_url" --connect-timeout 10 --max-time 20)
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Health check passed${NC}"
    echo "   📊 Response: $body" | head -c 100
    echo "..."
else
    echo -e "   ${RED}❌ Health check failed: HTTP $http_code${NC}"
    if [ "$http_code" = "000" ]; then
        echo -e "   ${YELLOW}💡 This might be expected if the function is cold-starting${NC}"
        echo -e "   ${YELLOW}   Modal functions scale to zero and need time to start up${NC}"
    fi
fi

# Test 2: Model info
echo -e "\n${BLUE}2. Testing info endpoint...${NC}"
info_url="${BASE_URL}-info.modal.run"
response=$(curl -s -w "%{http_code}" "$info_url" --connect-timeout 10 --max-time 20)
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Info endpoint working${NC}"
    echo "   📊 Model info retrieved successfully"
else
    echo -e "   ${YELLOW}⚠️  Info endpoint status: HTTP $http_code${NC}"
fi

# Test 3: Single prediction
echo -e "\n${BLUE}3. Testing single prediction...${NC}"
predict_url="${BASE_URL}-predict.modal.run"

# Test case 1: Setosa
echo "   🌸 Testing Setosa sample..."
response=$(curl -s -w "%{http_code}" -X POST "$predict_url" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  --connect-timeout 15 --max-time 30)
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Setosa prediction successful${NC}"
    predicted_class=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('class_name', 'unknown'))" 2>/dev/null)
    confidence=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('confidence', 'unknown'))" 2>/dev/null)
    echo "   📊 Prediction: $predicted_class (confidence: $confidence)"
else
    echo -e "   ${RED}❌ Setosa prediction failed: HTTP $http_code${NC}"
    echo "   Response: $body"

    if [ "$http_code" = "000" ]; then
        echo -e "   ${YELLOW}💡 Cold start detected. Modal functions may take 10-30 seconds on first request${NC}"
    fi
fi

# Test case 2: Versicolor (if first test succeeded)
if [ "$http_code" -eq 200 ]; then
    echo "   🌺 Testing Versicolor sample..."
    response=$(curl -s -w "%{http_code}" -X POST "$predict_url" \
      -H "Content-Type: application/json" \
      -d '{"features": [6.2, 2.8, 4.8, 1.8]}' \
      --connect-timeout 10 --max-time 15)
    http_code="${response: -3}"
    body="${response%???}"

    if [ "$http_code" -eq 200 ]; then
        echo -e "   ${GREEN}✅ Versicolor prediction successful${NC}"
        predicted_class=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('class_name', 'unknown'))" 2>/dev/null)
        confidence=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('confidence', 'unknown'))" 2>/dev/null)
        echo "   📊 Prediction: $predicted_class (confidence: $confidence)"
    else
        echo -e "   ${YELLOW}⚠️  Versicolor prediction: HTTP $http_code${NC}"
    fi
fi

# Test 4: Input validation
echo -e "\n${BLUE}4. Testing input validation...${NC}"
echo "   ⚠️  Testing invalid input (too few features)..."
response=$(curl -s -w "%{http_code}" -X POST "$predict_url" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5]}' \
  --connect-timeout 10 --max-time 15)
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -ne 200 ] || echo "$body" | grep -q "error"; then
    echo -e "   ${GREEN}✅ Input validation working (rejected invalid input)${NC}"
else
    echo -e "   ${YELLOW}⚠️  Input validation might not be working as expected${NC}"
fi

# Test 5: Batch prediction (if available)
echo -e "\n${BLUE}5. Testing batch prediction...${NC}"
batch_url="${BASE_URL}-batch-predict.modal.run"
response=$(curl -s -w "%{http_code}" -X POST "$batch_url" \
  -H "Content-Type: application/json" \
  -d '[{"features": [5.1, 3.5, 1.4, 0.2]}, {"features": [6.2, 2.8, 4.8, 1.8]}]' \
  --connect-timeout 10 --max-time 20)
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Batch prediction successful${NC}"
    count=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('count', 0))" 2>/dev/null)
    echo "   📊 Processed: $count samples"
else
    echo -e "   ${YELLOW}⚠️  Batch prediction: HTTP $http_code (may not be implemented)${NC}"
fi

# Performance info
echo -e "\n${PURPLE}📈 Modal Platform Benefits Demonstrated:${NC}"
echo "   🚀 Automatic scaling (including to zero when not in use)"
echo "   🌍 Global deployment with HTTPS endpoints"
echo "   💰 Pay-per-request pricing (no always-on costs)"
echo "   🔧 Zero infrastructure management required"
echo "   ⚡ Sub-second scaling for traffic spikes"
echo "   📊 Built-in monitoring and observability"

echo -e "\n${GREEN}🎉 Modal deployment test complete!${NC}"

echo -e "\n${CYAN}💡 Compare this to Parts 1-3:${NC}"
echo "   • No Docker files to maintain"
echo "   • No Kubernetes YAML configurations"
echo "   • No load balancer setup"
echo "   • No container registry management"
echo "   • No server provisioning or maintenance"
echo "   • No scaling configuration needed"
echo -e "\n${CYAN}   All of that complexity replaced with ~20 lines of Python! 🎯${NC}"
