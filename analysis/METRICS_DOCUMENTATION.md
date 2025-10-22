# Motion Blend Quality Metrics

## Overview

This module implements comprehensive quality metrics for evaluating blended motion sequences, following methodology from recent research literature.

## Metrics Implemented

### 1. L2 Velocity Metric

**Formula:** `Δv(t,j) = |v(t,j) - v(t-1,j)` where `v(t,j) = ||v(t,j)||₂`

Measures the difference in joint speed (L2 norm of velocity vector) between consecutive frames. This metric identifies discontinuities in motion smoothness.

**Interpretation:**
- Lower values indicate smoother motion
- Spikes indicate sudden changes in joint velocity
- Particularly useful for detecting artifacts in blend transition regions

**Key Joints Tracked:**
- Pelvis (root/hips)
- Left Wrist
- Right Wrist
- Left Foot
- Right Foot

### 2. L2 Acceleration Metric

**Formula:** `ΔΔv(t,j) = |Δv(t,j) - Δv(t-1,j)|`

Measures the temporal change in L2 velocity, corresponding to acceleration. This provides higher-order smoothness analysis.

**Interpretation:**
- Lower values indicate smoother acceleration profiles
- Detects jerky or abrupt motion changes
- Complements velocity metric for comprehensive smoothness assessment

### 3. Fréchet Inception Distance (FID)

Single-sample variant measuring distribution similarity between generated and real motions.

**Formula:** `FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real * Σ_gen)^(1/2))`

**Interpretation:**
- Lower is better (0-100+ scale)
- < 20: Excellent distribution match
- 20-40: Good quality
- > 40: Distribution mismatch

**Requirements:**
- Reference dataset of real motions
- Feature extractor (can use flattened positions or pretrained encoder)

### 4. Coverage (Cov)

Measures how well generated motions cover the real motion distribution.

**Interpretation:**
- Higher is better (0-1 scale)
- > 0.8: Excellent coverage
- 0.6-0.8: Good coverage
- < 0.6: Limited coverage

**Methodology:**
- For each real motion, check if any generated motion is in its k-nearest neighbors
- Reports fraction of real motions covered

### 5. Diversity Metrics

#### Global Diversity (GDiv)
Variance across entire motion sequence. Measures overall motion variety.

#### Local Diversity (LDiv)
Average variance in sliding windows (default 30 frames). Captures short-term motion variation.

#### Inter Diversity
Variance between different joints (spatial diversity). Measures how independently joints move.

#### Intra Diversity
Average variance within each joint's trajectory (temporal diversity). Measures consistency per joint.

**Interpretation:**
- Higher values generally indicate more expressive, varied motion
- Balance needed: too high may indicate noise, too low may be robotic
- Context-dependent (walking should be less diverse than dancing)

### 6. Transition Smoothness Score

Composite metric specifically evaluating the blend transition region.

**Components:**
- Velocity ratio: transition velocity / overall velocity
- Acceleration ratio: transition acceleration / overall acceleration
- Spike penalty: maximum velocity in transition region

**Formula:**
```
smoothness_score = 1.0 / (1.0 + velocity_ratio + acceleration_ratio + 0.1 * spike_penalty)
```

**Interpretation:**
- 0-1 scale, higher is better
- > 0.8: Smooth transition
- 0.6-0.8: Acceptable with minor artifacts
- < 0.6: Rough transition with visible discontinuities

### 7. Overall Quality Score

Weighted combination of all metrics:

```
quality_score = 0.4 * smoothness + 0.3 * diversity + 0.3 * fid_coverage
```

**Categories:**
- >= 0.80: Excellent
- 0.65-0.79: Good
- 0.50-0.64: Acceptable
- < 0.50: Poor

## Usage

### Basic Usage

```python
from motion_metrics import MotionData, MotionBlendEvaluator

# Load motion data
motion = MotionData(
    positions=positions_array,  # [frames, joints, 3]
    joint_names=['Hips', 'LeftWrist', ...],
    fps=30.0
)

# Create evaluator
evaluator = MotionBlendEvaluator(
    transition_window=(120, 180)  # Blend region frames
)

# Compute all metrics
results = evaluator.evaluate(motion, reference_motions=None)

# Access results
print(f"Quality Score: {results['quality_score']:.3f}")
print(f"Smoothness: {results['smoothness']['smoothness_score']:.3f}")
print(f"FID: {results['fid']}")
print(f"Coverage: {results['coverage']}")
```

### Command Line Tool

```bash
# Compute metrics for a blend file
python analysis/compute_blend_metrics.py \
    --blend-file data/blends/walk_run_blend_0.5.bvh \
    --transition-start 120 \
    --transition-end 180 \
    --output results/metrics.json \
    --visualize

# With BigQuery upload
python analysis/compute_blend_metrics.py \
    --blend-file data/blends/walk_run_blend_0.5.bvh \
    --transition-start 120 \
    --transition-end 180 \
    --upload-bigquery
```

### Visualization

The tool generates two types of visualizations:

1. **Velocity/Acceleration Plot**: Shows L2 velocity and acceleration over time for key joints, with transition region highlighted
2. **Summary Dashboard**: Comprehensive view of all metrics with bar charts and scores

## Integration with Pipeline

### 1. Compute Metrics (Python)

```bash
python analysis/compute_blend_metrics.py \
    --blend-file gs://bucket/blends/my_blend.bvh \
    --output metrics.json
```

### 2. Load to BigQuery (via Fivetran)

Metrics are ingested through the connector pipeline:
- Connector reads from GCS
- Transforms and loads to `RAW.blend_metrics` table
- dbt models aggregate in `RAW_marts.mart_blend_snn_metrics`

### 3. Query in dbt

```sql
-- Get high-quality blends
SELECT 
    blend_id,
    quality_score,
    quality_category,
    transition_smoothness,
    fid,
    coverage
FROM {{ ref('mart_blend_snn_metrics') }}
WHERE quality_category IN ('excellent', 'good')
ORDER BY quality_score DESC
LIMIT 100
```

### 4. Index in Elasticsearch

```bash
python exporter/bigquery_to_elastic.py \
    --bq-table RAW_marts.mart_blend_snn_metrics \
    --es-index mb_blends_v1
```

### 5. Search API

```python
# Find smooth high-coverage blends
GET /mb_blends_v1/_search
{
  "query": {
    "bool": {
      "must": [
        {"range": {"transition_smoothness": {"gte": 0.8}}},
        {"range": {"coverage": {"gte": 0.85}}},
        {"term": {"quality_category": "excellent"}}
      ]
    }
  },
  "sort": [{"quality_score": {"order": "desc"}}]
}
```

## References

1. **Tselepi et al. (2025)** - "Controllable Single-Shot Animation Blending with Temporal Conditioning"
   - L2 velocity/acceleration metrics
   - Transition smoothness evaluation
   - Key joint tracking methodology

2. **Guo et al. (2020)** - "Action2Motion: Generating Diverse and Natural Actions from the First Observation"
   - FID for motion evaluation
   - Coverage metric
   - Feature-based motion comparison

3. **Petrovich et al. (2021)** - "Action-Conditioned 3D Human Motion Synthesis with Transformers (ACTOR)"
   - Diversity metrics (Global, Local, Inter, Intra)
   - Motion quality assessment framework

4. **Heusel et al. (2017)** - "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
   - Original FID formulation
   - Fréchet distance between multivariate Gaussians

## Implementation Notes

### Dependencies

```txt
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

### Performance Considerations

- L2 velocity: O(frames × joints)
- L2 acceleration: O(frames × joints)
- Diversity: O(frames × joints × features)
- FID: O(N² × features) for distance matrix
- Coverage: O(N × M) for k-NN search

For large datasets (> 10K motions), use approximate nearest neighbors (e.g., FAISS, Annoy).

### Memory Requirements

- Motion data: frames × joints × 3 × 8 bytes (float64)
- 240 frames × 24 joints × 3 × 8 = ~138 KB per motion
- Batch processing recommended for > 1000 motions

## Future Enhancements

1. **Foot Skating Detection**: Measure unnatural foot sliding during motion
2. **Joint Angle Metrics**: Analyze anatomical plausibility
3. **Contact Preservation**: Verify hand/foot contacts maintained across blend
4. **Semantic Coherence**: Measure action label consistency
5. **Real-time Evaluation**: Optimize for sub-millisecond computation
6. **GPU Acceleration**: CUDA kernels for velocity/acceleration computation

## License

MIT License - See repository LICENSE file
