#!/bin/bash

# Script to run video content detection in Docker
# Usage: ./run_docker.sh [video_file] [additional_args...]

set -e

# Default values
VIDEO_FILE="${1:-input/movie.mp4}"
shift || true  # Remove first argument if it exists
ADDITIONAL_ARGS="$@"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Video Content Detection - Docker Runner${NC}"
echo "================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker and try again."
    exit 1
fi

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${YELLOW}⚠️  Warning: Video file '$VIDEO_FILE' not found${NC}"
    echo "Make sure your video file is in the input/ directory"
fi

# Check for GPU support
GPU_SUPPORT=""
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}🚀 NVIDIA GPU detected - enabling GPU acceleration${NC}"
        GPU_SUPPORT="--gpus all"
    else
        echo -e "${YELLOW}⚠️  NVIDIA drivers found but GPU not accessible${NC}"
    fi
else
    echo -e "${YELLOW}💻 No NVIDIA GPU detected - using CPU${NC}"
fi

# Build the image if it doesn't exist
if ! docker image inspect selections-visual:latest >/dev/null 2>&1; then
    echo -e "${BLUE}🔨 Building Docker image...${NC}"
    docker build -t selections-visual:latest .
else
    echo -e "${GREEN}✅ Docker image already exists${NC}"
fi

# Create output directory if it doesn't exist
mkdir -p output

# Run the container
echo -e "${BLUE}🎬 Processing video: $VIDEO_FILE${NC}"
echo "Additional args: $ADDITIONAL_ARGS"
echo ""

docker run --rm -it \
    $GPU_SUPPORT \
    -v "$(pwd)/input:/app/input:ro" \
    -v "$(pwd)/output:/app/output" \
    -e PYTHONUNBUFFERED=1 \
    selections-visual:latest \
    python visual_detection.py \
    --video_path "$VIDEO_FILE" \
    --device auto \
    --batch_size 8 \
    --mixed_precision \
    $ADDITIONAL_ARGS

echo ""
echo -e "${GREEN}✅ Processing complete!${NC}"
echo -e "${BLUE}📊 Results saved to: output/visual_detection_results.csv${NC}"
echo -e "${BLUE}🚀 GPU-optimized processing with mixed precision${NC}"
