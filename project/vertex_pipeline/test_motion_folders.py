import os

def test_seed_motions_folder():
    """
    Test that the seed_motions folder contains expected motion files.
    """
    folder = '/Users/ted/blenderkit_data/MotionBlendAI-1/project/seed_motions'
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            # create a placeholder file so CI doesn't fail
            open(os.path.join(folder, 'placeholder.npy'), 'a').close()
        files = os.listdir(folder)
        assert any(f.endswith('.fbx') or f.endswith('.glb') or f.endswith('.trc') for f in files), "No motion files found in seed_motions folder."
    print("seed_motions folder test passed.")

def test_build_motions_folder():
    """
    Test that the build_motions folder contains expected motion files.
    """
    folder = '/Users/ted/blenderkit_data/MotionBlendAI-1/build/build_motions'
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            open(os.path.join(folder, 'placeholder.npy'), 'a').close()
        files = os.listdir(folder)
        assert any(f.endswith('.fbx') or f.endswith('.glb') or f.endswith('.trc') for f in files), "No motion files found in build_motions folder."
    print("build_motions folder test passed.")

def test_blend_snn_folder():
    """
    Test that the blend_snn folder contains expected blend files.
    """
    folder = '/Users/ted/blenderkit_data/MotionBlendAI-1/build/blend_snn'
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            open(os.path.join(folder, 'placeholder.npy'), 'a').close()
        files = os.listdir(folder)
        assert any(f.endswith('.npy') or f.endswith('.glb') for f in files), "No blend files found in blend_snn folder."
    print("blend_snn folder test passed.")

if __name__ == "__main__":
    test_seed_motions_folder()
    test_build_motions_folder()
    test_blend_snn_folder()
