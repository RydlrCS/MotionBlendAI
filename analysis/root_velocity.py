"""
Root Velocity Calculations for Motion Analysis

Computes linear and angular velocities for root joint motion,
used in motion quality evaluation and feature extraction.

Based on: https://github.com/RydlrCS/blendanim
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "compute_linear_velocity_xz",
    "compute_angular_velocity",
    "compute_root_features"
]


def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    
    Args:
        q: First quaternion(s) [..., 4]
        r: Second quaternion(s) [..., 4]
    
    Returns:
        Product quaternion(s) [..., 4]
    """
    original_shape = q.shape
    q = q.reshape(-1, 4)
    r = r.reshape(-1, 4)
    
    # Quaternion multiplication: q * r
    w = q[:, 0] * r[:, 0] - q[:, 1] * r[:, 1] - q[:, 2] * r[:, 2] - q[:, 3] * r[:, 3]
    x = q[:, 0] * r[:, 1] + q[:, 1] * r[:, 0] + q[:, 2] * r[:, 3] - q[:, 3] * r[:, 2]
    y = q[:, 0] * r[:, 2] - q[:, 1] * r[:, 3] + q[:, 2] * r[:, 0] + q[:, 3] * r[:, 1]
    z = q[:, 0] * r[:, 3] + q[:, 1] * r[:, 2] - q[:, 2] * r[:, 1] + q[:, 3] * r[:, 0]
    
    result = np.stack([w, x, y, z], axis=-1)
    return result.reshape(original_shape)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate (inverse for unit quaternions).
    
    Args:
        q: Quaternion(s) [..., 4]
    
    Returns:
        Conjugate quaternion(s) [..., 4]
    """
    mask = np.ones_like(q)
    mask[..., 1:] = -1
    return q * mask


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector(s) by quaternion(s).
    
    Args:
        q: Quaternion(s) [..., 4]
        v: Vector(s) [..., 3]
    
    Returns:
        Rotated vector(s) [..., 3]
    """
    assert q.shape[-1] == 4, "Quaternion must have 4 components"
    assert v.shape[-1] == 3, "Vector must have 3 components"
    assert q.shape[:-1] == v.shape[:-1], "Shapes must match"
    
    original_shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    
    qvec = q[:, 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    
    result = v + 2 * (q[:, 0:1] * uv + uuv)
    return result.reshape(original_shape)


def compute_linear_velocity_xz(
    velocity: np.ndarray,
    rotations: np.ndarray
) -> np.ndarray:
    """
    Compute linear velocity in XZ plane (horizontal movement).
    
    Rotates velocity vectors to world space and extracts horizontal components.
    
    Args:
        velocity: Velocity vectors [T, 1, 3] or [T, 3]
        rotations: Root rotations as quaternions [T, 1, 4] or [T, 4]
    
    Returns:
        Horizontal velocity [T, 2] (X and Z components)
    """
    if velocity.ndim == 3:
        velocity = velocity[:, 0, :]  # [T, 3]
    if rotations.ndim == 3:
        rotations = rotations[:, 0, :]  # [T, 4]
    
    # Rotate velocity to world space
    velocity_world = _quat_rotate(rotations, velocity)
    
    # Extract XZ components
    return velocity_world[:, [0, 2]]


def compute_angular_velocity(
    rotations: np.ndarray,
    zero_pad: bool = True
) -> np.ndarray:
    """
    Compute angular velocity from rotation sequence.
    
    Args:
        rotations: Root rotations as quaternions [T, 4] or [T, J, 4]
        zero_pad: Whether to zero-pad first frame
    
    Returns:
        Angular velocity [T, 1] (Y-axis rotation)
    """
    if rotations.ndim == 3:
        # Multi-joint, extract root
        rotations = rotations[:, 0, :]  # [T, 4]
    
    T = rotations.shape[0]
    if T < 2:
        return np.zeros((T, 1))
    
    # Compute relative rotation between frames
    # r_velocity = q_t * conjugate(q_{t-1})
    r_velocity = _quat_multiply(
        rotations[1:],
        _quat_conjugate(rotations[:-1])
    )
    
    # Extract Y-axis rotation (arcsin of Z component)
    angular_vel = np.arcsin(r_velocity[:, 2:3])
    
    if zero_pad:
        # Pad first frame with zero
        pad = np.zeros((1, 1))
        angular_vel = np.concatenate([pad, angular_vel], axis=0)
    
    return angular_vel


def compute_root_features(
    position: np.ndarray,
    rotations: Optional[np.ndarray] = None,
    use_velocity: bool = True,
    keep_y_position: bool = True
) -> np.ndarray:
    """
    Compute root joint features for motion representation.
    
    Args:
        position: Root position [T, 3]
        rotations: Optional root rotations as quaternions [T, 4]
        use_velocity: Whether to compute velocities (vs. absolute positions)
        keep_y_position: Whether to keep Y position (vs. velocity)
    
    Returns:
        Root features array [T, D] where D depends on options
    """
    features = []
    
    if use_velocity:
        # Compute position velocities
        pos_vel = np.zeros_like(position)
        pos_vel[1:] = position[1:] - position[:-1]
        pos_vel[0] = pos_vel[1]  # Copy second frame to first
        
        if keep_y_position:
            # Use Y position, XZ velocity
            features.append(position[:, 1:2])  # Y position
            features.append(pos_vel[:, [0, 2]])  # XZ velocity
        else:
            # All velocity
            features.append(pos_vel)
    else:
        # Use absolute positions
        features.append(position)
    
    # Add angular velocity if rotations provided
    if rotations is not None:
        ang_vel = compute_angular_velocity(rotations, zero_pad=True)
        features.append(ang_vel)
    
    return np.concatenate(features, axis=-1)


def compute_velocity_statistics(
    positions: np.ndarray,
    fps: int = 30
) -> dict:
    """
    Compute velocity statistics for motion analysis.
    
    Args:
        positions: Joint positions [T, J, 3]
        fps: Frames per second
    
    Returns:
        Dictionary with velocity statistics
    """
    if positions.shape[0] < 2:
        return {
            'mean_velocity': 0.0,
            'max_velocity': 0.0,
            'velocity_variance': 0.0
        }
    
    # Compute frame-to-frame differences
    velocities = (positions[1:] - positions[:-1]) * fps  # Scale by fps
    
    # Compute magnitude per joint per frame
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)  # [T-1, J]
    
    return {
        'mean_velocity': float(np.mean(velocity_magnitudes)),
        'max_velocity': float(np.max(velocity_magnitudes)),
        'velocity_variance': float(np.var(velocity_magnitudes)),
        'mean_per_joint': velocity_magnitudes.mean(axis=0).tolist()
    }
