"""
Motion Quality Metrics for Blend Evaluation

Implements metrics from the blendanim repository for evaluating
the quality of generated motion blends:
- Coverage: How well the blend covers the space of real motions
- Global Diversity: Variety across entire sequences
- Local Diversity: Frame-to-frame variation
- L2 Velocity: Mean velocity magnitude
- L2 Acceleration: Mean acceleration magnitude

Based on: https://github.com/RydlrCS/blendanim
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Any
import logging

if TYPE_CHECKING:
    import numpy as np  # type: ignore[import]
    from numba import jit  # type: ignore[import]
else:
    try:
        import numpy as np
        from numba import jit
    except ImportError:
        np = None  # type: ignore
        jit = lambda **kwargs: lambda f: f  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "compute_coverage",
    "compute_global_diversity",
    "compute_local_diversity",
    "compute_l2_velocity",
    "compute_l2_acceleration",
    "evaluate_blend_quality"
]


@jit(nopython=True)  # type: ignore[misc]
def _nn_dp_kernel(G: Any, E: Any, F: Any, Cost: Any, tmin: int, L: int, Nt: int) -> Tuple[Any, Any, Any]:
    """
    Numba-optimized dynamic programming kernel for nearest neighbor search.
    """
    for i in range(L):
        for j in range(Nt):
            G[i, j] = Cost[i, j]
            E[i, j] = -1
            F[i, j] = -1
            if i >= tmin:
                for k in range(i - tmin + 1):
                    c = G[k, j - 1] + Cost[i, j]
                    if c < G[i, j]:
                        G[i, j] = c
                        E[i, j] = k
                        F[i, j] = j - 1
    return G, E, F


def _group_cost_from_arrays(src: Any, tgt: Any) -> Any:
    """
    Compute pairwise distance matrix between motion sequences.
    
    Args:
        src: Source motion [T1, D] where D is feature dimension
        tgt: Target motion [T2, D]
    
    Returns:
        Cost matrix [T1, T2]
    """
    # Compute L2 distances
    _t1, _d = src.shape
    _t2 = tgt.shape[0]
    
    # Expand dimensions for broadcasting
    src_expanded = src[:, np.newaxis, :]  # [T1, 1, D]
    tgt_expanded = tgt[np.newaxis, :, :]  # [1, T2, D]
    
    # Compute squared differences and sum over features
    diff = src_expanded - tgt_expanded
    cost = np.sqrt(np.sum(diff ** 2, axis=-1))  # [T1, T2]
    
    return cost


def compute_coverage(
    predicted: Any,  # np.ndarray
    ground_truth: Any,  # np.ndarray
    tmin: int = 30,
    threshold: float = 2.0
) -> float:
    """
    Compute coverage metric: how well predicted motions cover the space of ground truth.
    
    Coverage measures the percentage of time windows in the ground truth that have
    a similar counterpart in the predicted sequence.
    
    Args:
        predicted: Predicted motion sequence [T_pred, D]
        ground_truth: Ground truth motion sequence [T_gt, D]
        tmin: Minimum window size for matching
        threshold: Distance threshold for considering a match
    
    Returns:
        Coverage score in [0, 1]
    """
    if predicted.ndim == 3:
        # Flatten [T, J, D] -> [T, J*D]
        predicted = predicted.reshape(predicted.shape[0], -1)
        ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)
    
    group_cost = _group_cost_from_arrays(predicted, ground_truth)
    
    # Check coverage for each window
    results = []
    for i in range(group_cost.shape[0] - tmin):
        # Find minimum cost alignment for this window
        window_cost = group_cost[i, i + tmin]
        cost_per_frame = np.min(window_cost) / tmin
        results.append(1.0 if cost_per_frame < threshold else 0.0)
    
    return float(np.mean(results)) if results else 0.0


def compute_global_diversity(sequences: Any, tmin: int = 30) -> float:  # sequences: np.ndarray
    """
    Compute global diversity: variation across entire sequences.
    
    Args:
        sequences: Array of motion sequences [N, T, D] or [N, T, J, D]
        tmin: Minimum window size
    
    Returns:
        Global diversity score
    """
    if sequences.ndim == 4:
        # Flatten [N, T, J, D] -> [N, T, J*D]
        _n, _t, _j, _d = sequences.shape
        sequences = sequences.reshape(_n, _t, -1)
    
    n = len(sequences)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances between sequences
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            cost = _group_cost_from_arrays(sequences[i], sequences[j])
            # Use dynamic time warping distance
            mean_cost = np.mean(cost)
            distances.append(mean_cost)
    
    return float(np.mean(distances)) if distances else 0.0


def compute_local_diversity(sequence: Any, window_size: int = 30) -> float:  # sequence: np.ndarray
    """
    Compute local diversity: frame-to-frame variation within windows.
    
    Args:
        sequence: Motion sequence [T, D] or [T, J, D]
        window_size: Size of local window
    
    Returns:
        Local diversity score
    """
    if sequence.ndim == 3:
        # Flatten [T, J, D] -> [T, J*D]
        sequence = sequence.reshape(sequence.shape[0], -1)
    
    T, _d = sequence.shape
    if T < window_size:
        return 0.0
    
    # Compute frame-to-frame differences in windows
    diversities = []
    for i in range(0, T - window_size, window_size // 2):
        window = sequence[i:i + window_size]
        # Compute variance within window
        variance = np.var(window, axis=0)
        diversities.append(np.mean(variance))
    
    return float(np.mean(diversities)) if diversities else 0.0


def compute_l2_velocity(positions: Any, fps: int = 30) -> float:  # positions: np.ndarray
    """
    Compute mean L2 velocity magnitude.
    
    Args:
        positions: Joint positions [T, J, 3]
        fps: Frames per second
    
    Returns:
        Mean velocity in units per second
    """
    if positions.shape[0] < 2:
        return 0.0
    
    # Compute frame-to-frame differences
    dp = positions[1:] - positions[:-1]  # [T-1, J, 3]
    
    # Compute L2 norm per joint per frame
    velocities = np.linalg.norm(dp, axis=-1)  # [T-1, J]
    
    # Mean across joints and frames, scale by fps
    mean_velocity = np.mean(velocities) * fps
    
    return float(mean_velocity)


def compute_l2_acceleration(positions: Any, fps: int = 30) -> float:  # positions: np.ndarray
    """
    Compute mean L2 acceleration magnitude.
    
    Args:
        positions: Joint positions [T, J, 3]
        fps: Frames per second
    
    Returns:
        Mean acceleration in units per second^2
    """
    if positions.shape[0] < 3:
        return 0.0
    
    # Compute velocities
    velocities = positions[1:] - positions[:-1]  # [T-1, J, 3]
    
    # Compute accelerations (change in velocity)
    accelerations = velocities[1:] - velocities[:-1]  # [T-2, J, 3]
    
    # Compute L2 norm per joint per frame
    acc_magnitudes = np.linalg.norm(accelerations, axis=-1)  # [T-2, J]
    
    # Mean across joints and frames, scale by fps^2
    mean_acceleration = np.mean(acc_magnitudes) * (fps ** 2)
    
    return float(mean_acceleration)


def evaluate_blend_quality(
    blend: Any,  # np.ndarray
    reference: Optional[Any] = None,  # Optional[np.ndarray]
    fps: int = 30,
    compute_coverage_metric: bool = True
) -> Dict[str, float]:
    """
    Evaluate overall quality of a motion blend.
    
    Args:
        blend: Blended motion sequence [T, J, 3] or [T, D]
        reference: Optional reference motion for coverage comparison
        fps: Frames per second
        compute_coverage_metric: Whether to compute coverage (requires reference)
    
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Ensure blend is 3D for position-based metrics
    if blend.ndim == 2:
        # Assume [T, J*3] format, reshape to [T, J, 3]
        T, D = blend.shape
        if D % 3 == 0:
            J = D // 3
            blend_3d = blend.reshape(T, J, 3)
        else:
            logger.warning(f"Cannot reshape blend from [T, {D}] to [T, J, 3]")
            blend_3d = None
    else:
        blend_3d = blend
    
    # Compute velocity and acceleration if positions available
    if blend_3d is not None and blend_3d.shape[-1] == 3:
        metrics['l2_velocity_mean'] = compute_l2_velocity(blend_3d, fps)
        metrics['l2_acceleration_mean'] = compute_l2_acceleration(blend_3d, fps)
    
    # Compute local diversity
    metrics['local_diversity'] = compute_local_diversity(blend)
    
    # Compute coverage if reference provided
    if compute_coverage_metric and reference is not None:
        metrics['coverage'] = compute_coverage(blend, reference)
    
    # Compute quality score (weighted combination)
    # Higher is better for coverage and diversity, lower is better for velocity spikes
    quality_components = []
    
    if 'coverage' in metrics:
        quality_components.append(metrics['coverage'])
    
    if 'local_diversity' in metrics:
        # Normalize diversity (higher is better, but not too high)
        normalized_diversity = min(1.0, metrics['local_diversity'] / 10.0)
        quality_components.append(normalized_diversity)
    
    if 'l2_velocity_mean' in metrics:
        # Penalize very high velocities (likely artifacts)
        velocity_score = max(0.0, 1.0 - metrics['l2_velocity_mean'] / 50.0)
        quality_components.append(velocity_score)
    
    if quality_components:
        metrics['quality_score'] = float(np.mean(quality_components))
    else:
        metrics['quality_score'] = 0.0
    
    # Categorize quality
    score = metrics['quality_score']
    if score >= 0.85:
        metrics['quality_category'] = 'excellent'
    elif score >= 0.75:
        metrics['quality_category'] = 'good'
    elif score >= 0.65:
        metrics['quality_category'] = 'fair'
    else:
        metrics['quality_category'] = 'poor'
    
    return metrics
