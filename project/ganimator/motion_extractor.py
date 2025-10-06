def blend_from_similar(fbx_path, glb_files, trc_files):
    """
    If an FBX file is uploaded, find the most similar GLB or TRC file and blend from it.
    Returns the path to the best match and its type, or None if not found.
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
import os
import numpy as np
from typing import List, Optional
import shutil
import difflib

# --- SNN BlendNet Import ---
import sys
sys.path.append(os.path.dirname(__file__))
from snn_blendnet import blend_motion_snn

# --- FBX Extraction ---
def extract_fbx_joints(file_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint positions from an FBX file using the 'fbx' Python SDK (fbx2json fallback).
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        import fbx
        import FbxCommon
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
        joints = []
        def traverse(node):
            if node.GetNodeAttribute() and node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.eSkeleton:
                t = node.LclTranslation.Get()
                joints.append([t[0], t[1], t[2]])
            for i in range(node.GetChildCount()):
                traverse(node.GetChild(i))
        traverse(root)
        arr = np.array(joints)
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
        joints = []
        models = data.get('models', [])
        for model in models:
            if model.get('attrType') == 'LimbNode':
                t = model.get('Lcl Translation', [0, 0, 0])
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
        import trimesh
        scene = trimesh.load(file_path, force='scene')
        joints = []
        # Skip root/world nodes, collect all others
        for name in list(scene.graph.nodes):
            if name.lower() in ("world", "root", "scene"):
                continue
            try:
                matrix = scene.graph.get(name)[0]
                t = matrix[:3, 3]
                joints.append(t.tolist())
            except Exception as e:
                print(f"[GLB] Node {name} extraction error: {e}")
        arr = np.array(joints)
        return arr.reshape(1, arr.shape[0], 3) if arr.size > 0 else None
    except ImportError:
        print("[GLB] trimesh not installed. Skipping GLB extraction.")
        return None
    except Exception as e:
        print(f"[GLB] Error: {e}")
        return None


# --- TRC Extraction ---
def extract_trc_joints(file_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint positions from a TRC file (C3D marker format).
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Find the first line with only numbers (data start)
        data = []
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

    def parse_glb_filename(filename):
        # Example: Walk_WaveLeftHand.glb -> {'motion': 'Walk', 'gesture': 'WaveLeftHand'}
        base = os.path.splitext(filename)[0]
        parts = base.split('_', 1)
        if len(parts) == 2:
            return {'motion': parts[0], 'gesture': parts[1]}
        return {'motion': base, 'gesture': None}

    def parse_trc_filename(filename):
        # Example: Jump_Trial1.trc -> {'motion': 'Jump', 'trial': 'Trial1'}
        base = os.path.splitext(filename)[0]
        parts = base.split('_', 1)
        if len(parts) == 2:
            return {'motion': parts[0], 'trial': parts[1]}
        return {'motion': base, 'trial': None}

    def parse_fbx_filename(filename):
        # Example: Angry.fbx, Tennis_Match_Point.fbx, etc.
        base = os.path.splitext(filename)[0]
        # Try to infer structure: split by underscores, spaces, etc.
        parts = base.replace(' ', '_').split('_')
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
def blend_real_motions_snn(file1, file2, blend_weight=0.5):
    """
    Loads two motion files (GLB/TRC), extracts joints, blends using SNNBlendNet, prints result.
    """
    arr1 = extract_glb_joints(file1) if file1.lower().endswith('.glb') else extract_trc_joints(file1)
    arr2 = extract_glb_joints(file2) if file2.lower().endswith('.glb') else extract_trc_joints(file2)
    if arr1 is None or arr2 is None:
        print(f"[ERROR] Extraction failed for one or both files: {file1}, {file2}")
        return
    # Align shapes if possible (truncate to min joints)
    if arr1.shape != arr2.shape:
        print(f"[WARN] Shape mismatch: {arr1.shape} vs {arr2.shape}. Attempting to align by truncating to minimum joint count.")
        min_frames = min(arr1.shape[0], arr2.shape[0])
        min_joints = min(arr1.shape[1], arr2.shape[1])
        min_dim = min(arr1.shape[2], arr2.shape[2])
        arr1_aligned = arr1[:min_frames, :min_joints, :min_dim]
        arr2_aligned = arr2[:min_frames, :min_joints, :min_dim]
        if arr1_aligned.shape != arr2_aligned.shape or arr1_aligned.shape[1] < 3:
            print(f"[ERROR] Cannot align motions for blending. Aborting.")
            return
        arr1, arr2 = arr1_aligned, arr2_aligned
    blended = blend_motion_snn(arr1[0], arr2[0], blend_weight=blend_weight)
    print(f"[SNN BLEND] Blended shape: {blended.shape}")
    print(f"[SNN BLEND] First frame: {blended[0,0]}")
    # Save blended result as .npy in build/blend_snn
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/blend_snn'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]
    out_path = os.path.join(out_dir, f"blend_{base1}_{base2}.npy")
    np.save(out_path, blended)
    print(f"[SNN BLEND] Saved blended result to {out_path}")


# CLI usage: python motion_extractor.py blend_snn <file1> <file2> [blend_weight]
if len(sys.argv) > 1 and sys.argv[1] == 'blend_snn':
    if len(sys.argv) < 4:
        print("Usage: python motion_extractor.py blend_snn <file1> <file2> [blend_weight]")
        sys.exit(1)
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/build_motions'))
    def resolve_path(arg):
        # Use as-is if absolute or contains a directory, else prepend build_dir
        if os.path.isabs(arg) or os.path.dirname(arg):
            return os.path.abspath(arg)
        return os.path.join(build_dir, arg)
    file1 = resolve_path(sys.argv[2])
    file2 = resolve_path(sys.argv[3])
    blend_weight = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    if not os.path.exists(file1):
        print(f"[ERROR] File not found: {file1}")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"[ERROR] File not found: {file2}")
        sys.exit(1)
    blend_real_motions_snn(file1, file2, blend_weight)
def extract_trc_joints(file_path: str) -> Optional[np.ndarray]:
    """
    Extracts joint positions from a TRC file (text format).
    Returns a numpy array of shape (frames, joints, 3) or None if failed.
    """
    try:
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        # Find header and data start
        for i, line in enumerate(lines):
            if line.strip().startswith('Frame#'):
                header = lines[i]
                data_start = i + 1
                break
        else:
            return None
        # Parse data
        arr = []
        for line in lines[data_start:]:
            if not line.strip():
                continue
            vals = line.strip().split()
            # Skip frame/time columns, then group by 3 for x,y,z
            coords = [float(x) for x in vals[2:]]
            frame = [coords[j:j+3] for j in range(0, len(coords), 3)]
            arr.append(frame)
        return np.array(arr)  # (frames, joints, 3)
    except Exception as e:
        print(f"[TRC] Error: {e}")
        return None

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
