import os
import unittest
import difflib

SEED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../seed_motions'))
BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/build_motions'))

def get_motion_names_with_ext(folder, exts):
    return [os.path.splitext(f)[0] for f in os.listdir(folder) if any(f.lower().endswith(e) for e in exts)]

class TestFBXSimilarity(unittest.TestCase):
    def test_air_kicking_fbx_similarity(self):
        # Simulate an FBX file named 'Air Kicking_reprocessed.fbx'
        fbx_name = 'Air Kicking_reprocessed'
        # Find all GLB and TRC motion names
        motion_names = get_motion_names_with_ext(BUILD_DIR, ['.glb', '.trc'])
        # Use difflib to find the closest match
        matches = difflib.get_close_matches(fbx_name, motion_names, n=1, cutoff=0.6)
        self.assertTrue(matches, "No similar motion found for Air Kicking_reprocessed.fbx")
        # The closest match should be 'Air Kicking_mixamo' or 'Air Kicking_fist' (depending on files present)
        self.assertTrue(any('Air Kicking' in m for m in matches), f"Closest match does not contain 'Air Kicking': {matches}")

if __name__ == "__main__":
    unittest.main()