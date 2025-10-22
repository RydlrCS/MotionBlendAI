"""
Motion Blend Quality Metrics Implementation
============================================

Implements all metrics from relevant literature for motion blend evaluation:
- L2 Velocity: Δv(t,j) = |v(t,j) - v(t-1,j)| where v(t,j) = ||v(t,j)||₂
- L2 Acceleration: ΔΔv(t,j) = |Δv(t,j) - Δv(t-1,j)|
- Fréchet Inception Distance (FID)
- Coverage (Cov)
- Global Diversity (GDiv)
- Local Diversity (LDiv)
- Inter Diversity
- Intra Diversity

Key joints tracked: pelvis, left wrist, right wrist, left foot, right foot

References:
- Tselepi et al. (2025) "Controllable Single-Shot Animation Blending"
- Guo et al. (2020) "Action2Motion"
- Petrovich et al. (2021) "ACTOR"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import linalg


@dataclass
class MotionData:
    """Container for motion capture data"""
    positions: np.ndarray  # [frames, joints, 3]
    joint_names: List[str]
    fps: float = 30.0
    
    @property
    def frame_count(self) -> int:
        return self.positions.shape[0]
    
    @property
    def joint_count(self) -> int:
        return self.positions.shape[1]
    
    def get_joint_index(self, joint_name: str) -> Optional[int]:
        """Get index for joint name"""
        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            return None


class L2VelocityMetric:
    """
    Compute L2 velocity metric for motion smoothness
    
    Measures the difference in joint speed between consecutive frames.
    Formula: Δv(t,j) = |v(t,j) - v(t-1,j)| where v(t,j) = ||v(t,j)||₂
    """
    
    def __init__(self, key_joints: Optional[List[str]] = None):
        """
        Args:
            key_joints: List of joint names to track (default: pelvis, wrists, feet)
        """
        self.key_joints = key_joints or [
            'Hips', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot'
        ]
    
    def compute(self, motion: MotionData) -> Dict[str, np.ndarray]:
        """
        Compute L2 velocity for each joint
        
        Returns:
            Dictionary with:
            - 'per_joint': [frames-1, joints] array of L2 velocities
            - 'per_frame': [frames-1] average across joints
            - 'key_joints': [frames-1, len(key_joints)] for key joints only
        """
        # Step 1: Compute velocity vectors (position difference)
        velocity_vectors = np.diff(motion.positions, axis=0)  # [frames-1, joints, 3]
        
        # Step 2: Compute L2 norm of velocity vectors
        velocity_norms = np.linalg.norm(velocity_vectors, axis=2)  # [frames-1, joints]
        
        # Step 3: Compute L2 velocity (difference in velocity norms)
        l2_velocity = np.zeros_like(velocity_norms)
        l2_velocity[1:] = np.abs(np.diff(velocity_norms, axis=0))
        
        # Extract key joints
        key_joint_indices = [
            motion.get_joint_index(j) for j in self.key_joints 
            if motion.get_joint_index(j) is not None
        ]
        key_joints_velocity = l2_velocity[:, key_joint_indices] if key_joint_indices else l2_velocity
        
        return {
            'per_joint': l2_velocity,
            'per_frame': np.mean(l2_velocity, axis=1),
            'key_joints': key_joints_velocity,
            'joint_names': [motion.joint_names[i] for i in key_joint_indices]
        }


class L2AccelerationMetric:
    """
    Compute L2 acceleration metric for higher-order smoothness
    
    Measures the change in L2 velocity over time (temporal acceleration).
    Formula: ΔΔv(t,j) = |Δv(t,j) - Δv(t-1,j)|
    """
    
    def __init__(self, key_joints: Optional[List[str]] = None):
        self.key_joints = key_joints or [
            'Hips', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot'
        ]
    
    def compute(self, l2_velocity: np.ndarray, motion: MotionData) -> Dict[str, np.ndarray]:
        """
        Compute L2 acceleration from L2 velocity
        
        Args:
            l2_velocity: Output from L2VelocityMetric ['per_joint']
            motion: Original motion data for joint indices
            
        Returns:
            Dictionary with:
            - 'per_joint': [frames-2, joints] array
            - 'per_frame': [frames-2] average across joints
            - 'key_joints': [frames-2, len(key_joints)]
        """
        # Compute temporal change in L2 velocity
        l2_acceleration = np.abs(np.diff(l2_velocity, axis=0))
        
        # Extract key joints
        key_joint_indices = [
            motion.get_joint_index(j) for j in self.key_joints 
            if motion.get_joint_index(j) is not None
        ]
        key_joints_accel = l2_acceleration[:, key_joint_indices] if key_joint_indices else l2_acceleration
        
        return {
            'per_joint': l2_acceleration,
            'per_frame': np.mean(l2_acceleration, axis=1),
            'key_joints': key_joints_accel,
            'joint_names': [motion.joint_names[i] for i in key_joint_indices]
        }


class TransitionSmoothnessMetric:
    """
    Evaluate smoothness specifically in the blend transition region
    
    Lower values indicate smoother transitions with fewer discontinuities
    """
    
    def __init__(self, transition_window: Tuple[int, int]):
        """
        Args:
            transition_window: (start_frame, end_frame) for blend region
        """
        self.transition_start, self.transition_end = transition_window
    
    def compute(self, l2_velocity: np.ndarray, l2_acceleration: np.ndarray) -> Dict[str, float]:
        """
        Compute smoothness metrics for transition region
        
        Returns:
            Dictionary with smoothness scores and statistics
        """
        # Extract transition region
        trans_velocity = l2_velocity[self.transition_start:self.transition_end]
        trans_accel = l2_acceleration[self.transition_start:self.transition_end]
        
        # Compute statistics
        trans_vel_mean = np.mean(trans_velocity)
        trans_vel_max = np.max(trans_velocity)
        trans_accel_mean = np.mean(trans_accel)
        trans_accel_max = np.max(trans_accel)
        
        # Compare to overall sequence
        overall_vel_mean = np.mean(l2_velocity)
        overall_accel_mean = np.mean(l2_acceleration)
        
        # Smoothness ratios (lower is better)
        velocity_ratio = trans_vel_mean / (overall_vel_mean + 1e-8)
        acceleration_ratio = trans_accel_mean / (overall_accel_mean + 1e-8)
        
        # Combined smoothness score (0-1, higher is better)
        # Penalize spikes in transition region
        spike_penalty = trans_vel_max / (overall_vel_mean + 1e-8)
        smoothness_score = 1.0 / (1.0 + velocity_ratio + acceleration_ratio + 0.1 * spike_penalty)
        
        return {
            'smoothness_score': float(smoothness_score),
            'velocity_ratio': float(velocity_ratio),
            'acceleration_ratio': float(acceleration_ratio),
            'transition_velocity_mean': float(trans_vel_mean),
            'transition_acceleration_mean': float(trans_accel_mean),
            'transition_velocity_max': float(trans_vel_max),
            'overall_velocity_mean': float(overall_vel_mean),
            'overall_acceleration_mean': float(overall_accel_mean)
        }


class DiversityMetrics:
    """
    Compute diversity metrics: Global, Local, Inter, and Intra
    
    These metrics assess the variety and expressiveness of motion
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Window size for local diversity computation
        """
        self.window_size = window_size
    
    def compute(self, motion: MotionData) -> Dict[str, float]:
        """
        Compute all diversity metrics
        
        Returns:
            Dictionary with GDiv, LDiv, Inter, and Intra diversity
        """
        positions = motion.positions  # [frames, joints, 3]
        
        # Flatten for global analysis
        positions_flat = positions.reshape(motion.frame_count, -1)
        
        # Global Diversity: variance across entire sequence
        global_div = float(np.var(positions_flat))
        
        # Local Diversity: average variance in sliding windows
        local_vars = []
        for i in range(0, len(positions_flat) - self.window_size, self.window_size):
            window = positions_flat[i:i+self.window_size]
            local_vars.append(np.var(window))
        local_div = float(np.mean(local_vars)) if local_vars else 0.0
        
        # Inter Diversity: variance between different joints (spatial)
        joint_means = np.mean(positions, axis=0)  # [joints, 3]
        inter_div = float(np.var(joint_means))
        
        # Intra Diversity: average variance within each joint trajectory (temporal)
        intra_vars = []
        for j in range(motion.joint_count):
            joint_traj = positions[:, j, :]
            intra_vars.append(np.var(joint_traj))
        intra_div = float(np.mean(intra_vars))
        
        return {
            'global_diversity': global_div,
            'local_diversity': local_div,
            'inter_diversity': inter_div,
            'intra_diversity': intra_div
        }


class FIDCoverageMetrics:
    """
    Compute Fréchet Inception Distance (FID) and Coverage (Cov)
    
    FID measures distribution similarity between generated and real motions
    Coverage measures how well generated motions cover the real distribution
    
    Requires a reference dataset for comparison
    """
    
    def __init__(self, feature_extractor: Optional[callable] = None):
        """
        Args:
            feature_extractor: Function to extract features from motion
                              (e.g., pretrained motion encoder)
        """
        self.feature_extractor = feature_extractor
    
    def compute_fid(self, real_features: np.ndarray, generated_features: np.ndarray) -> float:
        """
        Compute Fréchet Inception Distance
        
        FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real * Σ_gen)^(1/2))
        
        Args:
            real_features: [N, D] features from real motions
            generated_features: [M, D] features from generated motions
            
        Returns:
            FID score (lower is better)
        """
        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(generated_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(generated_features, rowvar=False)
        
        # Compute FID
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        return float(fid)
    
    def compute_coverage(self, real_features: np.ndarray, generated_features: np.ndarray, 
                        k: int = 3) -> float:
        """
        Compute Coverage metric
        
        Measures the fraction of real motions that have at least one generated 
        motion within their k-nearest neighbors
        
        Args:
            real_features: [N, D] features from real motions
            generated_features: [M, D] features from generated motions
            k: Number of nearest neighbors
            
        Returns:
            Coverage score (higher is better, 0-1)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Fit k-NN on combined dataset
        combined = np.vstack([real_features, generated_features])
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(combined)
        
        # For each real sample, check if any generated sample is in k-NN
        covered = 0
        n_real = len(real_features)
        
        for i in range(n_real):
            distances, indices = nbrs.kneighbors([real_features[i]])
            # Check if any generated sample (index >= n_real) is in neighbors
            if np.any(indices[0][1:] >= n_real):  # Skip first (self)
                covered += 1
        
        coverage = covered / n_real
        return float(coverage)
    
    def compute(self, motion: MotionData, reference_motions: List[MotionData]) -> Dict[str, float]:
        """
        Compute FID and Coverage for a single motion against reference dataset
        
        Args:
            motion: Generated motion to evaluate
            reference_motions: List of real reference motions
            
        Returns:
            Dictionary with FID and Coverage scores
        """
        if self.feature_extractor is None:
            # Use simple flattened positions as features if no encoder provided
            gen_features = motion.positions.reshape(motion.frame_count, -1)
            real_features = np.vstack([
                m.positions.reshape(m.frame_count, -1) for m in reference_motions
            ])
        else:
            gen_features = self.feature_extractor(motion.positions)
            real_features = np.vstack([
                self.feature_extractor(m.positions) for m in reference_motions
            ])
        
        fid = self.compute_fid(real_features, gen_features)
        coverage = self.compute_coverage(real_features, gen_features)
        
        return {
            'fid': fid,
            'coverage': coverage
        }


class MotionBlendEvaluator:
    """
    Complete evaluator for motion blend quality
    
    Combines all metrics into a single evaluation pipeline
    """
    
    def __init__(self, transition_window: Tuple[int, int], 
                 key_joints: Optional[List[str]] = None):
        """
        Args:
            transition_window: (start_frame, end_frame) for blend region
            key_joints: List of joint names to track
        """
        self.transition_window = transition_window
        self.key_joints = key_joints or [
            'Hips', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot'
        ]
        
        # Initialize metric computers
        self.l2_velocity = L2VelocityMetric(key_joints=self.key_joints)
        self.l2_acceleration = L2AccelerationMetric(key_joints=self.key_joints)
        self.smoothness = TransitionSmoothnessMetric(transition_window)
        self.diversity = DiversityMetrics(window_size=30)
        self.fid_cov = FIDCoverageMetrics()
    
    def evaluate(self, motion: MotionData, 
                 reference_motions: Optional[List[MotionData]] = None) -> Dict:
        """
        Perform complete evaluation of motion blend
        
        Args:
            motion: Motion to evaluate
            reference_motions: Optional reference dataset for FID/Coverage
            
        Returns:
            Dictionary with all computed metrics
        """
        # Compute L2 velocity
        velocity_results = self.l2_velocity.compute(motion)
        
        # Compute L2 acceleration
        accel_results = self.l2_acceleration.compute(
            velocity_results['per_joint'], motion
        )
        
        # Compute transition smoothness
        smoothness_results = self.smoothness.compute(
            velocity_results['per_joint'],
            accel_results['per_joint']
        )
        
        # Compute diversity metrics
        diversity_results = self.diversity.compute(motion)
        
        # Compute FID and Coverage if reference data provided
        if reference_motions:
            fid_cov_results = self.fid_cov.compute(motion, reference_motions)
        else:
            fid_cov_results = {'fid': None, 'coverage': None}
        
        # Aggregate all results
        return {
            'l2_velocity': {
                'mean': float(np.mean(velocity_results['per_joint'])),
                'std': float(np.std(velocity_results['per_joint'])),
                'max': float(np.max(velocity_results['per_joint'])),
                'per_frame': velocity_results['per_frame'].tolist(),
                'key_joints_mean': float(np.mean(velocity_results['key_joints']))
            },
            'l2_acceleration': {
                'mean': float(np.mean(accel_results['per_joint'])),
                'std': float(np.std(accel_results['per_joint'])),
                'max': float(np.max(accel_results['per_joint'])),
                'per_frame': accel_results['per_frame'].tolist(),
                'key_joints_mean': float(np.mean(accel_results['key_joints']))
            },
            'smoothness': smoothness_results,
            'diversity': diversity_results,
            'fid': fid_cov_results['fid'],
            'coverage': fid_cov_results['coverage'],
            'quality_score': self._compute_quality_score(
                smoothness_results, diversity_results, fid_cov_results
            )
        }
    
    def _compute_quality_score(self, smoothness: Dict, diversity: Dict, 
                               fid_cov: Dict) -> float:
        """
        Compute overall quality score (0-1, higher is better)
        
        Combines smoothness, diversity, and FID/Coverage into single metric
        """
        # Smoothness component (0-1, from smoothness_score)
        smooth_score = smoothness['smoothness_score']
        
        # Diversity component (normalize global diversity)
        div_score = min(1.0, diversity['global_diversity'] / 10.0)
        
        # FID/Coverage component (if available)
        if fid_cov['fid'] is not None and fid_cov['coverage'] is not None:
            # FID: lower is better, normalize to 0-1
            fid_score = 1.0 / (1.0 + fid_cov['fid'] / 50.0)
            # Coverage: higher is better, already 0-1
            cov_score = fid_cov['coverage']
            fid_cov_score = (fid_score + cov_score) / 2.0
        else:
            fid_cov_score = 0.5  # Neutral if not available
        
        # Weighted combination
        quality = 0.4 * smooth_score + 0.3 * div_score + 0.3 * fid_cov_score
        
        return float(quality)
