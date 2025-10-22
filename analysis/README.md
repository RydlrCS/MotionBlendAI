# Motion Blend Analysis Suite

Complete quantitative and qualitative evaluation framework for Rydlr Moverse motion blending experiments.

## 🎯 Overview

This analysis suite implements the methodology from **Tselepi et al. (2025)** "Controllable Single-Shot Animation Blending with Temporal Conditioning" for evaluating motion blend quality.

**Experiments:**
1. `Punches_Air Kicking_fist_blend_0.50`
2. `Tai_Chi_Flow Yoga_Pose_Flow_blend_0.50`

**Pipeline:** Rydlr Moverse → Fivetran → Elasticsearch

## 📊 Metrics Computed

### Quantitative Metrics

1. **L2 Velocity** - Measures joint speed differences:
   ```
   Δv(t,j) = |v(t,j) - v(t-1,j)| where v(t,j) = ||v(t,j)||₂
   ```

2. **L2 Acceleration** - Measures temporal velocity changes:
   ```
   ΔΔv(t,j) = |Δv(t,j) - Δv(t-1,j)|
   ```

3. **Fréchet Inception Distance (FID)** - Distribution similarity (lower is better)

4. **Coverage (Cov)** - Motion space coverage (higher is better)

5. **Diversity Metrics:**
   - Global Diversity (GDiv) - Overall sequence variance
   - Local Diversity (LDiv) - Windowed variance
   - Inter Diversity - Variance between joints
   - Intra Diversity - Variance within joint trajectories

### Qualitative Analysis

- Side-by-side velocity/acceleration visualizations
- Transition window highlighting (frames 120-180)
- 5-joint tracking: Pelvis, LeftWrist, RightWrist, LeftFoot, RightFoot
- Frame-by-frame comparison grids

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /Users/ted/blenderkit_data/MotionBlendAI-1
source .venv/bin/activate
pip install torch numpy matplotlib seaborn pandas
```

### 2. Run Analysis (Command Line)

```bash
# Single blend
python3 analysis/analyse_blend.py --blend "Punches_Air Kicking_fist_blend_0.50"

# Or use the automated runner
./run_analysis.sh
```

### 3. Run Analysis (Jupyter Notebook)

```bash
jupyter notebook analysis/motion_blend_analysis.ipynb
```

## 📁 Project Structure

```
MotionBlendAI-1/
├── analysis/
│   ├── analyse_blend.py          # Main analysis script
│   └── motion_blend_analysis.ipynb  # Jupyter notebook
├── data/
│   └── blends/                   # BVH blend files
├── outputs/
│   ├── analysis/                 # Visualizations (PNG/MP4)
│   └── metrics/                  # JSON metric files
└── run_analysis.sh               # Automated runner
```

## 🔬 Usage Examples

### Analyze Single Blend

```python
from analyse_blend import analyse_blend

metrics = analyse_blend(
    blend_name="Punches_Air Kicking_fist_blend_0.50",
    data_dir="./data/blends",
    output_dir="./outputs/analysis"
)

print(f"Smoothness: {metrics['transition_smoothness']:.4f}")
print(f"FID: {metrics['fid']:.4f}")
```

### Load and Visualize

```python
from analyse_blend import BVHMotionLoader, MotionMetricsCalculator
import matplotlib.pyplot as plt

# Load motion
loader = BVHMotionLoader("data/blends/Punches_Air Kicking_fist_blend_0.50.bvh")
motion_data = loader.load()

# Calculate metrics
calculator = MotionMetricsCalculator(motion_data, transition_window=(120, 180))
l2_velocity = calculator.compute_l2_velocity()

# Plot
plt.plot(l2_velocity)
plt.axvline(120, linestyle='--', label='Transition Start')
plt.axvline(180, linestyle='--', label='Transition End')
plt.legend()
plt.show()
```

### Upload to Fivetran

```python
from analyse_blend import FivetranUploader

uploader = FivetranUploader(endpoint="http://localhost:5000/api/metrics")
success = uploader.upload_metrics("blend_name", metrics)
```

## 📈 Output Files

### Metrics JSON
```json
{
  "blend_id": "Punches_Air Kicking_fist_blend_0.50",
  "timestamp": "2025-10-14T09:30:00.000Z",
  "metrics": {
    "transition_smoothness": 0.9234,
    "fid": 12.34,
    "coverage": 0.87,
    "diversity": {
      "global_diversity": 0.456,
      "local_diversity": 0.378
    }
  }
}
```

### Visualizations
- `{blend_name}_metrics.png` - Velocity/acceleration graphs
- `comparative_metrics.csv` - Side-by-side comparison table

## 🖥️ Hardware Requirements

**Minimum:**
- CPU: Intel or AMD x86_64
- RAM: 8GB
- Storage: 2GB for outputs

**Recommended:**
- GPU: NVIDIA RTX 4090 or equivalent
- RAM: 16GB
- CUDA: 11.7+

**Training Time:** ~3 hours per 360-frame sequence on RTX 4090

## 🔗 Integration

### Elasticsearch Schema

The analysis uploads metrics to the `moverse_blend_metrics` table:

```json
{
  "mappings": {
    "properties": {
      "blend_id": { "type": "keyword" },
      "timestamp": { "type": "date" },
      "transition_smoothness": { "type": "float" },
      "fid": { "type": "float" },
      "coverage": { "type": "float" },
      "diversity": { "type": "object" }
    }
  }
}
```

### API Endpoints

- `POST /api/metrics` - Upload blend metrics
- `GET /api/metrics/{blend_id}` - Retrieve metrics
- `POST /api/refresh_index` - Refresh Elasticsearch index

## 📚 References

1. **Tselepi et al. (2025)** - "Controllable Single-Shot Animation Blending with Temporal Conditioning"
2. **Perez et al. (2018)** - "FiLM: Visual Reasoning with a General Conditioning Layer" (ArXiv:1709.07871)
3. **Heusel et al. (2017)** - "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID metric)

## 🐛 Troubleshooting

### BVH Files Not Found
```bash
# Check data directory
ls data/blends/

# Create mock data for testing
mkdir -p data/blends
# Place your BVH files here
```

### Import Errors
```bash
pip install -r requirements.txt
```

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # GPU name
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

Rydlr - GitHub: @RydlrCS
