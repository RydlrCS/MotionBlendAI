""" 
BVH Export Utilities for Motion Blends

Writes generated motion blends to BVH (BioVision Hierarchy) format
for use in animation software and further processing.

Based on: https://github.com/RydlrCS/blendanim
"""

from typing import TYPE_CHECKING, Sequence, List, Any
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    import numpy as np  # type: ignore[import]
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = ["save_bvh", "BvhJoint"]


@dataclass
class BvhJoint:
    """Represents a joint in the BVH hierarchy."""
    id: int
    parent_id: int
    children: List['BvhJoint']


def _create_hierarchy(parents: Any) -> BvhJoint:  # parents: np.ndarray
    """
    Create hierarchical joint structure from parent indices.
    
    Args:
        parents: Array of parent indices, where parents[i] is the parent of joint i
    
    Returns:
        Root joint of the hierarchy
    """
    joints = [BvhJoint(id=i, parent_id=int(parents[i]), children=[]) for i in range(len(parents))]
    
    # Build hierarchy
    root = None
    for joint in joints:
        if joint.parent_id == -1:
            root = joint
        else:
            parent = joints[joint.parent_id]
            parent.children.append(joint)
    
    if root is None:
        raise ValueError("No root joint found (expected parent_id == -1)")
    return root


def _add_to_parent(pid: int, cid: int, joint: BvhJoint) -> bool:
    """Recursively add child to parent joint."""
    if joint.id == pid:
        joint.children.append(BvhJoint(id=cid, parent_id=pid, children=[]))
        return True
    
    for child in joint.children:
        if _add_to_parent(pid, cid, child):
            return True
    
    return False


def _write_joint(
    file: Any,
    joint: BvhJoint,
    ordered_ids: List[int],
    names: Sequence[str],
    offsets: Any,  # np.ndarray
    indent: int,
    precision: int
) -> None:
    """
    Recursively write joint hierarchy to BVH file.
    
    Args:
        file: Open file handle
        joint: Current joint to write
        ordered_ids: List to collect joint IDs in BVH order
        names: Joint names
        offsets: Joint offsets [J, 3]
        indent: Current indentation level
        precision: Decimal precision for floats
    """
    is_root = joint.id == 0
    offset = offsets[joint.id]
    name = names[joint.id]
    ordered_ids.append(joint.id)
    
    tab = "\t"
    
    # Write joint header
    file.write(f'{tab*indent}{"ROOT" if is_root else "JOINT"} {name}\n')
    file.write(f"{tab*indent}{{\n")
    
    # Write offset
    file.write(f"{tab*(indent+1)}OFFSET ")
    file.write(f"{round(offset[0], precision)} ")
    file.write(f"{round(offset[1], precision)} ")
    file.write(f"{round(offset[2], precision)}\n")
    
    # Write channels
    if is_root:
        file.write(
            f"{tab*(indent+1)}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
        )
    else:
        file.write(f"{tab*(indent+1)}CHANNELS 3 Xrotation Yrotation Zrotation\n")
    
    # Recursively write children
    for child in joint.children:
        _write_joint(file, child, ordered_ids, names, offsets, indent + 1, precision)
    
    # Write end site for leaf joints
    if len(joint.children) == 0:
        file.write(f"{tab*(indent+1)}End Site\n")
        file.write(f"{tab*(indent+1)}{{\n")
        file.write(f"{tab*(indent+2)}OFFSET 0.0 0.0 0.0\n")
        file.write(f"{tab*(indent+1)}}}\n")
    
    file.write(f"{tab*indent}}}\n")


def _write_motion(
    file: Any,
    rotations: Any,  # np.ndarray
    position: Any,  # np.ndarray
    precision: int
) -> None:
    """
    Write motion data (positions and rotations) to BVH file.
    
    Args:
        file: Open file handle
        rotations: Joint rotations in Euler angles [T, J, 3]
        position: Root position [T, 3]
        precision: Decimal precision for floats
    """
    for pos, rots in zip(position, rotations):
        # Write root position
        file.write(
            f"{round(pos[0], precision)} {round(pos[1], precision)} {round(pos[2], precision)} "
        )
        
        # Write joint rotations
        for rot in rots:
            file.write(
                f"{round(rot[0], precision)} {round(rot[1], precision)} {round(rot[2], precision)} "
            )
        
        file.write("\n")


def save_bvh(
    path: str,
    frames: int,
    timestep: float,
    names: Sequence[str],
    parents: Any,  # np.ndarray
    offsets: Any,  # np.ndarray
    rotations: Any,  # np.ndarray
    position: Any,  # np.ndarray
    precision: int = 6
) -> None:
    """
    Save motion data to BVH file format.
    
    Args:
        path: Output file path
        frames: Number of frames
        timestep: Time between frames (e.g., 1/30 for 30 FPS)
        names: Joint names [J]
        parents: Parent indices [J]
        offsets: Joint offsets from parents [J, 3]
        rotations: Joint rotations in Euler angles (degrees) [T, J, 3]
        position: Root position [T, 3]
        precision: Decimal precision for output values
    
    Example:
        >>> save_bvh(
        ...     "output.bvh",
        ...     frames=120,
        ...     timestep=1/30,
        ...     names=["Hips", "Spine", "LeftLeg", ...],
        ...     parents=np.array([-1, 0, 0, ...]),
        ...     offsets=np.array([[0, 0, 0], [0, 5, 0], ...]),
        ...     rotations=np.random.randn(120, 24, 3),
        ...     position=np.random.randn(120, 3)
        ... )
    """
    # Create hierarchy
    root = _create_hierarchy(parents)
    
    with open(path, "w") as file:
        # Write hierarchy section
        file.write("HIERARCHY\n")
        ordered_ids = []
        _write_joint(file, root, ordered_ids, names, offsets, 0, precision)
        
        # Write motion section
        file.write("MOTION\n")
        file.write(f"Frames: {frames}\n")
        file.write(f"Frame Time: {timestep}\n")
        
        # Reorder rotations according to BVH hierarchy
        rotations_ordered = rotations[:, ordered_ids, :]
        
        _write_motion(file, rotations_ordered, position, precision)
        file.write("\n")
    
    logger.info(f"Saved BVH file to {path} ({frames} frames, {len(names)} joints)")


def create_demo_bvh(path: str, frames: int = 120, fps: int = 30) -> None:
    """
    Create a simple demo BVH file for testing.
    
    Args:
        path: Output file path
        frames: Number of frames
        fps: Frames per second
    """
    # Simple skeleton: Hips -> Spine -> Chest
    names = ["Hips", "Spine", "Chest", "Neck", "Head"]
    parents = np.array([-1, 0, 1, 2, 3])
    offsets = np.array([
        [0, 0, 0],      # Hips
        [0, 10, 0],     # Spine
        [0, 15, 0],     # Chest
        [0, 20, 0],     # Neck
        [0, 5, 0]       # Head
    ])
    
    # Generate simple animation (slight rotation over time)
    timestep = 1.0 / fps
    rotations = np.zeros((frames, len(names), 3))
    for t in range(frames):
        angle = (t / frames) * 360
        rotations[t, 1, 1] = angle * 0.1  # Spine Y rotation
        rotations[t, 2, 0] = np.sin(angle * np.pi / 180) * 10  # Chest X rotation
    
    # Simple root movement
    position = np.zeros((frames, 3))
    for t in range(frames):
        position[t, 0] = np.sin(t / frames * 2 * np.pi) * 5
        position[t, 2] = t * 0.1
    
    save_bvh(path, frames, timestep, names, parents, offsets, rotations, position)
    logger.info(f"Created demo BVH at {path}")
