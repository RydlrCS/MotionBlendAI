#!/bin/bash
# Motion Blend Analysis Runner
# Automates quantitative analysis for multiple blend experiments

echo "🚀 Motion Blend Analysis Pipeline"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Run: source .venv/bin/activate"
    exit 1
fi

# Check Python dependencies
python -c "import torch; import numpy; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install torch numpy matplotlib seaborn pandas
fi

# Create output directories
mkdir -p outputs/analysis
mkdir -p outputs/metrics
mkdir -p data/blends

echo "✅ Environment ready"
echo ""

# GPU utilization logging (if CUDA available)
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "🖥️  CUDA detected - logging GPU utilization"
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 5 > outputs/gpu_utilization.log &
    GPU_PID=$!
    echo "   GPU monitoring PID: $GPU_PID"
fi

# Run analyses
echo "📊 Running blend analyses..."
echo ""

echo "1️⃣  Analyzing: Punches_Air Kicking_fist_blend_0.50"
python3 analysis/analyse_blend.py --blend "Punches_Air Kicking_fist_blend_0.50" \
    --data-dir "data/blends" \
    --output-dir "outputs/analysis"

if [ $? -eq 0 ]; then
    echo "✅ Blend 1 complete"
else
    echo "❌ Blend 1 failed"
fi

echo ""
echo "2️⃣  Analyzing: Tai_Chi_Flow Yoga_Pose_Flow_blend_0.50"
python3 analysis/analyse_blend.py --blend "Tai_Chi_Flow Yoga_Pose_Flow_blend_0.50" \
    --data-dir "data/blends" \
    --output-dir "outputs/analysis"

if [ $? -eq 0 ]; then
    echo "✅ Blend 2 complete"
else
    echo "❌ Blend 2 failed"
fi

# Stop GPU monitoring
if [ ! -z "$GPU_PID" ]; then
    kill $GPU_PID 2>/dev/null
    echo ""
    echo "📈 GPU utilization log: outputs/gpu_utilization.log"
fi

# Optional: Trigger Elasticsearch index refresh
echo ""
echo "🔄 Refreshing Elasticsearch index..."
curl -X POST http://localhost:5000/api/refresh_index \
    -H "Content-Type: application/json" \
    -d '{"index": "moverse_blend_metrics"}' \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Index refreshed"
else
    echo "⚠️  Index refresh failed (Flask may not be running)"
fi

echo ""
echo "=================================="
echo "✅ Analysis pipeline complete!"
echo ""
echo "📁 Outputs:"
echo "   - Metrics: outputs/metrics/"
echo "   - Analysis: outputs/analysis/"
echo "   - GPU logs: outputs/gpu_utilization.log"
echo ""
echo "📊 Next steps:"
echo "   1. Review Jupyter notebook: analysis/motion_blend_analysis.ipynb"
echo "   2. Check Elasticsearch: http://localhost:5000/api/metrics"
echo "   3. Generate visualizations with the notebook"
