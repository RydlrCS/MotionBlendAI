from typing import Optional, Tuple, Literal, List, Any, Dict
import os

def blend_from_similar(fbx_path: str, glb_files: List[str], trc_files: List[str]) -> Tuple[Optional[str], Optional[Literal['GLB', 'TRC']]]:
    """
    If an FBX file is uploaded, find the most similar GLB or TRC file and blend from it.
    Returns the path to the best match and its type, or (None, None) if not found.
    """
    import difflib
    fbx_name = os.path.splitext(os.path.basename(fbx_path))[0]
    all_motion_files = glb_files + trc_files
    all_motion_names = [os.path.splitext(os.path.basename(f))[0] for f in all_motion_files]
    matches = difflib.get_close_matches(fbx_name, all_motion_names, n=1, cutoff=0.6)
    if matches:
        best_match = matches[0]
        for f in glb_files:
            if os.path.splitext(os.path.basename(f))[0] == best_match:
                print(f"[BLEND] Blending from GLB: {f}")
                return f, 'GLB'
        for f in trc_files:
            if os.path.splitext(os.path.basename(f))[0] == best_match:
                print(f"[BLEND] Blending from TRC: {f}")
                return f, 'TRC'
    print(f"[BLEND] No similar GLB or TRC found for {fbx_path}")
    return None, None
"""
motion_extractor.py
Extracts joint data from FBX, BVH, and TRC files for use in motion blending and ML pipelines.
Uses open-source libraries: 'fbx', 'bvh', and 'numpy'.
"""
import numpy as np
from typing import List, Optional
import shutil

# --- SNN BlendNet Import ---
import sys
sys.path.append(os.path.dirname(__file__))

from typing import Optional, Callable

# Attempt to import the runtime implementation, but expose a typed wrapper so static analysis
# tools know the exact signature of blend_motion_snn.
# Declare a typed placeholder so static type checkers understand the expected signature.
_blend_motion_snn_impl: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None
try:
    from snn_blendnet import blend_motion_snn as _blend_motion_snn_impl  # type: ignore
except Exception:
    _blend_motion_snn_impl = None

# Import numpy here for type annotations; importing twice is harmless.
import numpy as np

def blend_motion_snn(src_arr: np.ndarray, tgt_arr: np.ndarray, blend_weight: float = 0.5) -> np.ndarray:
    """
    Typed wrapper around snn_blendnet.blend_motion_snn to provide accurate typing for static analysis
    and a clear runtime error if the implementation is unavailable.
    """
    if _blend_motion_snn_impl is None:
        raise ImportError("snn_blendnet.blend_motion_snn is not available")
    # Narrow the optional implementation using a runtime assertion, then call it directly.
    assert _blend_motion_snn_impl is not None
    impl: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = _blend_motion_snn_impl
    return impl(src_arr, tgt_arr, blend_weight)

# --- FBX Extraction ---
def extract_fbx_joints(file_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint positions from an FBX file using the 'fbx' Python SDK (fbx2json fallback).
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        # Import at runtime using importlib to avoid static linter errors when the fbx SDK
        # is not installed in the development environment.
        import importlib
        try:
            fbx = importlib.import_module('fbx')
            FbxCommon = importlib.import_module('FbxCommon')
        except Exception:
            print("[FBX] fbx SDK not installed. Skipping FBX extraction.")
            return None

        manager, scene = FbxCommon.InitializeSdkObjects()
        result = FbxCommon.LoadScene(manager, scene, file_path)
        if not result:
            print(f"[FBX] Failed to load {file_path}")
            return None
        # Traverse skeleton and extract joint positions for each frame
        # (Simplified: just extract first frame for demo)
        root = scene.GetRootNode()
        if not root:
            return None
        joints: List[List[float]] = []
        def traverse(node: Any) -> None:
            if node.GetNodeAttribute() and node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.eSkeleton:
                t = node.LclTranslation.Get()
                # ensure numeric types for numpy and static typing
                joints.append([float(t[0]), float(t[1]), float(t[2])])
            # Ensure we have an integer child count for range() to satisfy static type checkers.
            try:
                child_count = int(node.GetChildCount())
            except Exception:
                child_count = 0
            for i in range(child_count):
                child = node.GetChild(i)
                if child is not None:
                    traverse(child)
        traverse(root)
        arr = np.asarray(joints, dtype=np.float64)
        if arr.size == 0:
            return None
        return arr.reshape(1, arr.shape[0], 3)  # (frames=1, joints, 3)
    except ImportError:
        print("[FBX] fbx SDK not installed. Skipping FBX extraction.")
        return None
    except Exception as e:
        print(f"[FBX] Error: {e}")
        return None

# --- FBX2JSON Extraction ---
import json
def extract_fbx2json_joints(json_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint positions from a fbx2json-converted JSON file.
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    if not os.path.exists(json_path):
        print(f"[FBX2JSON] File not found: {json_path}")
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # This assumes the JSON structure contains a 'models' list with 'Lcl Translation' for each joint per frame
        # You may need to adjust this logic based on your actual fbx2json output
        joints: List[List[float]] = []
        models = data.get('models', [])
        for model in models:
            if model.get('attrType') == 'LimbNode':
                t_raw = model.get('Lcl Translation', [0, 0, 0])
                try:
                    t = [float(v) for v in t_raw]
                except Exception:
                    # Fallback to a zero translation if conversion fails or unexpected structure
                    t = [0.0, 0.0, 0.0]
                joints.append(t)
        arr = np.array(joints)
        return arr.reshape(1, arr.shape[0], 3)  # (frames=1, joints, 3)
    except Exception as e:
        print(f"[FBX2JSON] Error: {e}")
        return None



# --- GLB Extraction ---
def extract_glb_joints(file_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint positions from a GLB (glTF binary) file.
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        # Use importlib to import trimesh at runtime so static analyzers don't flag a hard import.
        import importlib
        try:
            trimesh = importlib.import_module('trimesh')
        except Exception:
            print("[GLB] trimesh not installed. Skipping GLB extraction.")
            return None

        scene = trimesh.load(file_path, force='scene')
        joints: List[List[float]] = []
        # Skip root/world nodes, collect all others
        for name in list(scene.graph.nodes):
            if name.lower() in ("world", "root", "scene"):
                continue
            try:
                matrix = scene.graph.get(name)[0]
                t = matrix[:3, 3]
                # ensure explicit float values so type checkers see a List[List[float]]
                joints.append([float(t[0]), float(t[1]), float(t[2])])
            except Exception as e:
                print(f"[GLB] Node {name} extraction error: {e}")
        arr = np.array(joints)
        return arr.reshape(1, arr.shape[0], 3) if arr.size > 0 else None
    except Exception as e:
        print(f"[GLB] Error: {e}")
        return None


# --- BVH Extraction ---
def extract_bvh_joints(file_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint positions from a BVH (Biovision Hierarchy) file.
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        # Use importlib to import bvh at runtime so static analyzers don't flag a hard import.
        import importlib
        try:
            bvh = importlib.import_module('bvh')
        except Exception:
            print("[BVH] bvh library not installed. Skipping BVH extraction.")
            return None

        with open(file_path, 'r') as f:
            mocap = bvh.Bvh(f.read())
        
        # Extract joint positions for all frames
        joint_names = mocap.get_joints_names()
        frames_data: List[List[List[float]]] = []
        
        for frame_i in range(mocap.nframes):
            frame_joints: List[List[float]] = []
            for joint_name in joint_names:
                try:
                    # Get joint position for this frame
                    pos = mocap.frame_joint_channels(frame_i, joint_name, ['Xposition', 'Yposition', 'Zposition'])
                    frame_joints.append([float(pos[0]), float(pos[1]), float(pos[2])])
                except Exception:
                    # If position channels don't exist, use zeros
                    frame_joints.append([0.0, 0.0, 0.0])
            frames_data.append(frame_joints)
        
        arr = np.array(frames_data)
        return arr if arr.size > 0 else None
    except Exception as e:
        print(f"[BVH] Error: {e}")
        return None

# --- TRC Extraction (legacy numeric-detection parser) ---
def extract_trc_joints_legacy(file_path: str) -> Optional[np.ndarray]:
    """
    Legacy TRC parser: extracts joint positions from a TRC file by detecting numeric-only data lines.
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Find the first line with only numbers (data start)
        data: List[List[List[float]]] = []
        for line in lines:
            parts = line.strip().split()
            # Skip empty lines and lines with non-numeric data
            if not parts or not all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts[2:]):
                continue
            try:
                row = [float(x) for x in parts[2:]]
                frame = [row[j:j+3] for j in range(0, len(row), 3)]
                data.append(frame)
            except Exception:
                continue
        arr = np.array(data)
        return arr if arr.size > 0 else None
    except Exception as e:
        print(f"[TRC] Error: {e}")
        return None

def extract_trc_joints(file_path: str) -> Optional[np.ndarray]:
    """
    Unified TRC extractor: tries header-based TRC parsing first (Frame# style), then falls back to the
    legacy numeric-detection parser (extract_trc_joints_legacy) if no header is found.
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        # Try header-based parsing (common text TRC format with 'Frame#' header)
        try:
            with open(file_path, 'r', errors='ignore') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip().startswith('Frame#'):
                    data_start = i + 1
                    arr: List[List[List[float]]] = []
                    for line2 in lines[data_start:]:
                        if not line2.strip():
                            continue
                        vals = line2.strip().split()
                        # Skip frame/time columns, then group by 3 for x,y,z
                        try:
                            coords = [float(x) for x in vals[2:]]
                        except Exception:
                            # Malformed row, skip
                            continue
                        frame = [coords[j:j+3] for j in range(0, len(coords), 3)]
                        arr.append(frame)
                    result = np.array(arr)
                    return result if result.size > 0 else None
        except Exception:
            # Fall through to legacy parser if header-based parsing fails
            pass

        # Fallback to legacy numeric-detection parser
        return extract_trc_joints_legacy(file_path)
    except Exception as e:
        print(f"[TRC] Error: {e}")
        return None

# --- Example usage for FBX2JSON extraction ---
if __name__ == "__main__":
    import sys
    import glob
    import subprocess

    # Ensure output directory exists
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/build_motions'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    seed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../seed_motions'))
    print(f"Scanning {seed_dir} for GLB, FBX (via JSON), and TRC files...")

    glb_files = [f for f in glob.glob(os.path.join(seed_dir, '*.glb')) if not os.path.basename(f).startswith('.')]
    trc_files = [f for f in glob.glob(os.path.join(seed_dir, '*.trc')) if not os.path.basename(f).startswith('.')]
    fbx_files = [f for f in glob.glob(os.path.join(seed_dir, '*.fbx')) if not os.path.basename(f).startswith('.')]

    # Copy GLB and TRC files to output_dir for indexing
    for file_list in [glb_files, trc_files]:
        for src_path in file_list:
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} -> {dst_path}")
            else:
                print(f"Already indexed: {dst_path}")

    def parse_glb_filename(filename: str) -> Dict[str, Optional[str]]:
        # Example: Walk_WaveLeftHand.glb -> {'motion': 'Walk', 'gesture': 'WaveLeftHand'}
        base = os.path.splitext(filename)[0]
        parts = base.split('_', 1)
        if len(parts) == 2:
            return {'motion': parts[0], 'gesture': parts[1]}
        return {'motion': base, 'gesture': None}

    def parse_trc_filename(filename: str) -> Dict[str, Optional[str]]:
        # Example: Jump_Trial1.trc -> {'motion': 'Jump', 'trial': 'Trial1'}
        base = os.path.splitext(filename)[0]
        parts = base.split('_', 1)
        if len(parts) == 2:
            return {'motion': parts[0], 'trial': parts[1]}
        return {'motion': base, 'trial': None}

    def parse_fbx_filename(filename: str) -> Dict[str, Any]:
        # Example: Angry.fbx, Tennis_Match_Point.fbx, etc.
        base = os.path.splitext(filename)[0]
        # Try to infer structure: split by underscores, spaces, etc.
        parts: List[str] = base.replace(' ', '_').split('_')
        return {'parts': parts, 'raw': base}

    for glb_path in glb_files:
        info = parse_glb_filename(os.path.basename(glb_path))
        arr = extract_glb_joints(glb_path)
        if arr is not None:
            print(f"GLB: {os.path.basename(glb_path)} | motion: {info['motion']} | gesture: {info['gesture']} | shape: {arr.shape}")
        else:
            print(f"GLB: {os.path.basename(glb_path)} | motion: {info['motion']} | gesture: {info['gesture']} | extraction failed")

    for trc_path in trc_files:
        info = parse_trc_filename(os.path.basename(trc_path))
        arr = extract_trc_joints(trc_path)
        if arr is not None:
            print(f"TRC: {os.path.basename(trc_path)} | motion: {info['motion']} | trial: {info['trial']} | shape: {arr.shape}")
        else:
            print(f"TRC: {os.path.basename(trc_path)} | motion: {info['motion']} | trial: {info['trial']} | extraction failed")

    for fbx_path in fbx_files:
        # Try to blend from a similar GLB or TRC file
        blend_from_similar(fbx_path, glb_files, trc_files)
        info = parse_fbx_filename(os.path.basename(fbx_path))
        json_path = fbx_path + '.json'
        if not os.path.exists(json_path):
            print(f"Converting {os.path.basename(fbx_path)} to JSON...")
            try:
                subprocess.run(['fbx2json', fbx_path], check=True)
            except Exception as e:
                print(f"[ERROR] Failed to convert {fbx_path}: {e}")
                continue
        else:
            print(f"JSON already exists for {os.path.basename(fbx_path)}")
        arr = extract_fbx2json_joints(json_path)
        if arr is not None:
            print(f"FBX: {os.path.basename(fbx_path)} | parts: {info['parts']} | shape: {arr.shape}")
        else:
            print(f"FBX: {os.path.basename(fbx_path)} | parts: {info['parts']} | extraction failed")


# --- SNN Blend Real Motions CLI ---
def blend_real_motions_snn(file1: str, file2: str, blend_weight: float = 0.5) -> None:
    """
    Loads two motion files (GLB/TRC), extracts joints, blends using SNNBlendNet, prints result.
    """
    arr1: Optional[np.ndarray] = extract_glb_joints(file1) if file1.lower().endswith('.glb') else extract_trc_joints(file1)
    arr2: Optional[np.ndarray] = extract_glb_joints(file2) if file2.lower().endswith('.glb') else extract_trc_joints(file2)
    if arr1 is None or arr2 is None:
        print(f"[ERROR] Extraction failed for one or both files: {file1}, {file2}")
        return
    # Align shapes if possible (truncate to min joints)
    if arr1.shape != arr2.shape:
        print(f"[WARN] Shape mismatch: {arr1.shape} vs {arr2.shape}. Attempting to align by truncating to minimum joint count.")
        min_frames = min(arr1.shape[0], arr2.shape[0])
        min_joints = min(arr1.shape[1], arr2.shape[1])
        min_dim = min(arr1.shape[2], arr2.shape[2])
        arr1 = arr1[:min_frames, :min_joints, :min_dim]
        arr2 = arr2[:min_frames, :min_joints, :min_dim]
    
    # Blend the motion arrays
    try:
        blended = blend_motion_snn(arr1, arr2, blend_weight)
        print(f"[SNN] Blended shape: {blended.shape}")
        print(f"[SNN] Sample output (first frame, first 3 joints):\n{blended[0, :3, :]}")
    except Exception as e:
        print(f"[ERROR] Blending failed: {e}")

# --- Unified API ---
def extract_motion(file_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint data from FBX, BVH, or TRC file.
    Returns a numpy array (frames, joints, 3) or None.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.fbx':
        return extract_fbx_joints(file_path)
    elif ext == '.bvh':
        return extract_bvh_joints(file_path)
    elif ext == '.trc':
        return extract_trc_joints(file_path)
    else:
        print(f"[MotionExtractor] Unsupported file type: {file_path}")
        return None

if __name__ == '__main__':
    # Demo: extract from all files in a folder
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else 'project/seed_motions'
    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        arr = extract_motion(path)
        if arr is not None:
            print(f"{fn}: shape {arr.shape}")
        else:
            print(f"{fn}: extraction failed")
