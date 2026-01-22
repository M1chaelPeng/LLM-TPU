#!/bin/bash

# Qwen3-VL Web Demo Launch Script

# Default paths
MODEL_PATH="${MODEL_PATH:-../config}"
BMODEL_PATH="${BMODEL_PATH:-}"

# Check if bmodel path is provided
if [ -z "$BMODEL_PATH" ]; then
    echo "Error: Please set BMODEL_PATH environment variable"
    echo "Usage: BMODEL_PATH=/path/to/model.bmodel $0"
    echo ""
    echo "Example:"
    echo "  export BMODEL_PATH=/path/to/qwen3-vl-4b.bmodel"
    echo "  ./run_demo.sh"
    exit 1
fi

# Check if bmodel file exists
if [ ! -f "$BMODEL_PATH" ]; then
    echo "Error: Model file not found: $BMODEL_PATH"
    exit 1
fi

echo "Starting Qwen3-VL Web Demo..."
echo "Model Path: $BMODEL_PATH"
echo "Config Path: $MODEL_PATH"
echo ""

# Launch web demo
python3 web_demo_gradio.py \
    --model_path "$BMODEL_PATH" \
    --config_path "$MODEL_PATH" \
    --devid 0 \
    --video_ratio 0.25
