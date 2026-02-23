#!/bin/bash
# Forgetless API Test Script
# Tests all API scenarios with real PDFs

set -e

BASE_URL="${BASE_URL:-http://localhost:8080}"
PDF_DIR="benches/data/large"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

header() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
}

test_result() {
    local name=$1
    local result=$2

    if echo "$result" | grep -q '"stats"'; then
        local input=$(echo "$result" | grep -o '"input_tokens":[0-9]*' | cut -d: -f2)
        local output=$(echo "$result" | grep -o '"output_tokens":[0-9]*' | cut -d: -f2)
        local ratio=$(echo "$result" | grep -o '"compression_ratio":[0-9.]*' | cut -d: -f2)
        local time=$(echo "$result" | grep -o '"processing_time_ms":[0-9]*' | cut -d: -f2)
        local chunks=$(echo "$result" | grep -o '"chunks_processed":[0-9]*' | cut -d: -f2)
        local selected=$(echo "$result" | grep -o '"chunks_selected":[0-9]*' | cut -d: -f2)

        echo -e "${GREEN}✓ $name${NC}"
        echo "  Input:  ${input} tokens"
        echo "  Output: ${output} tokens"
        echo "  Compression: ${ratio}x"
        echo "  Chunks: ${chunks} → ${selected}"
        echo "  Time: ${time}ms"
        return 0
    else
        echo -e "${RED}✗ $name${NC}"
        echo "  Error: Invalid response"
        echo "  Response: $(echo "$result" | head -c 200)"
        return 1
    fi
}

# Check server
header "Checking Server"
if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Server running at $BASE_URL${NC}"
else
    echo -e "${RED}✗ Server not running${NC}"
    echo "Start with: cargo run --features server,metal --release --bin forgetless-server"
    exit 1
fi

# Test 1: Health
header "Test 1: Health Check"
RESULT=$(curl -sf "$BASE_URL/health")
if [ "$RESULT" == "ok" ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed: $RESULT${NC}"
fi

# Test 2: Simple text
header "Test 2: Simple Text"
RESULT=$(curl -s --max-time 30 -X POST "$BASE_URL/" \
    -F 'metadata={"max_tokens": 1000, "contents": [{"content": "Hello world, this is a test.", "priority": "high"}]};type=application/json')
test_result "Simple text" "$RESULT"

# Test 3: Multiple contents with priorities
header "Test 3: Multiple Contents"
RESULT=$(curl -s --max-time 30 -X POST "$BASE_URL/" \
    -F 'metadata={"max_tokens": 5000, "query": "Main topic?", "contents": [{"content": "SYSTEM: You are an AI.", "priority": "critical"}, {"content": "User wants to learn about ML.", "priority": "high"}, {"content": "Background info.", "priority": "low"}]};type=application/json')
test_result "Multiple contents" "$RESULT"

# Test 4: Single PDF
header "Test 4: Single PDF"
PDF=$(ls $PDF_DIR/*.pdf | head -1)
echo "  File: $(basename $PDF)"
RESULT=$(curl -s --max-time 60 -X POST "$BASE_URL/" \
    -F 'metadata={"max_tokens": 32000, "query": "Summarize"};type=application/json' \
    -F "file=@$PDF")
test_result "Single PDF" "$RESULT"

# Test 5: 3 PDFs
header "Test 5: 3 PDFs"
PDFS=$(ls $PDF_DIR/*.pdf | head -3)
for f in $PDFS; do echo "  File: $(basename $f)"; done
CMD="curl -s --max-time 120 -X POST '$BASE_URL/' -F 'metadata={\"max_tokens\": 64000};type=application/json'"
for f in $PDFS; do CMD="$CMD -F 'file=@$f'"; done
RESULT=$(eval $CMD)
test_result "3 PDFs" "$RESULT"

# Test 6: 5 PDFs with query
header "Test 6: 5 PDFs with Query"
PDFS=$(ls $PDF_DIR/*.pdf | head -5)
CMD="curl -s --max-time 180 -X POST '$BASE_URL/' -F 'metadata={\"max_tokens\": 64000, \"query\": \"Compare approaches\"};type=application/json'"
for f in $PDFS; do CMD="$CMD -F 'file=@$f'"; done
RESULT=$(eval $CMD)
test_result "5 PDFs with query" "$RESULT"

# Test 7: Mixed priorities
header "Test 7: Mixed Content with Priorities"
PDFS=$(ls $PDF_DIR/*.pdf | head -3)
CMD="curl -s --max-time 120 -X POST '$BASE_URL/' -F 'metadata={\"max_tokens\": 32000, \"contents\": [{\"content\": \"Focus on innovations.\", \"priority\": \"critical\"}]};type=application/json'"
i=0
for f in $PDFS; do
    case $i in
        0) CMD="$CMD -F 'file:high=@$f'" ;;
        1) CMD="$CMD -F 'file:medium=@$f'" ;;
        *) CMD="$CMD -F 'file:low=@$f'" ;;
    esac
    ((i++))
done
RESULT=$(eval $CMD)
test_result "Mixed priorities" "$RESULT"

# Test 8: Aggressive compression
header "Test 8: Aggressive Compression (→ 8K)"
PDFS=$(ls $PDF_DIR/*.pdf | head -5)
CMD="curl -s --max-time 180 -X POST '$BASE_URL/' -F 'metadata={\"max_tokens\": 8000};type=application/json'"
for f in $PDFS; do CMD="$CMD -F 'file=@$f'"; done
RESULT=$(eval $CMD)
test_result "Aggressive compression" "$RESULT"

# Summary
header "Test Summary"
echo -e "${GREEN}All tests completed!${NC}"
echo ""
echo "Run 'kill \$(pgrep forgetless-server)' to stop the server."
