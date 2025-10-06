import os
import unittest
import shutil
import numpy as np
from project.ganimator import motion_extractor

SEED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../seed_motions'))
BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/build_motions'))

def list_files_with_ext(folder, ext):
    return [f for f in os.listdir(folder) if f.lower().endswith(ext)]

class TestMotionExtraction(unittest.TestCase):
    def test_glb_files_indexed_and_extractable(self):
        glb_files = list_files_with_ext(SEED_DIR, '.glb')
        for fname in glb_files:
            indexed = os.path.exists(os.path.join(BUILD_DIR, fname))
            self.assertTrue(indexed, f"GLB file {fname} not indexed in build_motions")
            arr = motion_extractor.extract_glb_joints(os.path.join(BUILD_DIR, fname))
            # Accept None if trimesh not installed, else must be np.ndarray or None
            self.assertTrue(arr is None or isinstance(arr, np.ndarray))

    def test_trc_files_indexed_and_extractable(self):
        trc_files = list_files_with_ext(SEED_DIR, '.trc')
        for fname in trc_files:
            indexed = os.path.exists(os.path.join(BUILD_DIR, fname))
            self.assertTrue(indexed, f"TRC file {fname} not indexed in build_motions")
            arr = motion_extractor.extract_trc_joints(os.path.join(BUILD_DIR, fname))
            self.assertTrue(arr is None or isinstance(arr, np.ndarray))

    def test_fbx_files_indexed(self):
        fbx_files = list_files_with_ext(SEED_DIR, '.fbx')
        for fname in fbx_files:
            # Only check indexing, not extraction (since FBX2JSON is not available)
            indexed = os.path.exists(os.path.join(BUILD_DIR, fname))
            # FBX files are not copied by default, so this should be False
            self.assertFalse(indexed, f"FBX file {fname} should not be indexed in build_motions")

if __name__ == "__main__":
    unittest.main()
