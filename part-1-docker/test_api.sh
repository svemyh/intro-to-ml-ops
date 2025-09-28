#!/bin/bash
# Test script for the Iris Classification API using curl commands

BASE_URL=${1:-"http://localhost:8000"}
echo "🧪 Testing API at $BASE_URL"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Health check
echo -e "\n${BLUE}1. Testing health endpoint...${NC}"
response=$(curl -s -w "%{http_code}" "$BASE_URL/health")
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Health check passed${NC}"
    echo "   📊 Response: $body" | head -c 100
else
    echo -e "   ${RED}❌ Health check failed: HTTP $http_code${NC}"
fi

# Test 2: Root endpoint
echo -e "\n${BLUE}2. Testing root endpoint...${NC}"
response=$(curl -s -w "%{http_code}" "$BASE_URL/")
http_code="${response: -3}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Root endpoint working${NC}"
else
    echo -e "   ${RED}❌ Root endpoint failed: HTTP $http_code${NC}"
fi

# Test 3: Valid predictions
echo -e "\n${BLUE}3. Testing predictions with valid data...${NC}"

# Test case 1: Setosa
echo "   🌸 Testing Setosa sample..."
response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}')
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Setosa prediction successful${NC}"
    echo "   📊 Response: $body"
else
    echo -e "   ${RED}❌ Setosa prediction failed: HTTP $http_code${NC}"
    echo "   Response: $body"
fi

# Test case 2: Versicolor
echo "   🌺 Testing Versicolor sample..."
response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [6.2, 2.8, 4.8, 1.8]}')
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Versicolor prediction successful${NC}"
    echo "   📊 Response: $body"
else
    echo -e "   ${RED}❌ Versicolor prediction failed: HTTP $http_code${NC}"
fi

# Test case 3: Virginica
echo "   🌻 Testing Virginica sample..."
response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [7.3, 2.9, 6.3, 1.8]}')
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo -e "   ${GREEN}✅ Virginica prediction successful${NC}"
    echo "   📊 Response: $body"
else
    echo -e "   ${RED}❌ Virginica prediction failed: HTTP $http_code${NC}"
fi

# Test 4: Invalid input validation
echo -e "\n${BLUE}4. Testing input validation...${NC}"

# Too few features
echo "   ⚠️  Testing too few features..."
response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5]}')
http_code="${response: -3}"

if [ "$http_code" -ne 200 ]; then
    echo -e "   ${GREEN}✅ Correctly rejected too few features (HTTP $http_code)${NC}"
else
    echo -e "   ${RED}❌ Should have rejected too few features${NC}"
fi

# Too many features
echo "   ⚠️  Testing too many features..."
response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2, 1.0]}')
http_code="${response: -3}"

if [ "$http_code" -ne 200 ]; then
    echo -e "   ${GREEN}✅ Correctly rejected too many features (HTTP $http_code)${NC}"
else
    echo -e "   ${RED}❌ Should have rejected too many features${NC}"
fi

echo -e "\n${GREEN}🎉 API testing complete!${NC}"