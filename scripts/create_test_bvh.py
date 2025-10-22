"""
Create minimal mock BVH files for testing the pipeline
These are valid BVH files with minimal data
"""
import os

# Minimal BVH template with 3 joints, 10 frames
MINIMAL_BVH = """HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT LeftUpLeg
    {
        OFFSET 3.90 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.00 -17.00 0.00
        }
    }
    JOINT RightUpLeg
    {
        OFFSET -3.90 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.00 -17.00 0.00
        }
    }
}
MOTION
Frames: 10
Frame Time: 0.033333
0.00 39.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
0.00 39.10 0.00 0.50 0.00 0.00 0.50 0.00 0.00 -0.50 0.00 0.00
0.00 39.20 0.00 1.00 0.00 0.00 1.00 0.00 0.00 -1.00 0.00 0.00
0.00 39.30 0.00 1.50 0.00 0.00 1.50 0.00 0.00 -1.50 0.00 0.00
0.00 39.40 0.00 2.00 0.00 0.00 2.00 0.00 0.00 -2.00 0.00 0.00
0.00 39.50 0.00 2.50 0.00 0.00 2.50 0.00 0.00 -2.50 0.00 0.00
0.00 39.60 0.00 3.00 0.00 0.00 3.00 0.00 0.00 -3.00 0.00 0.00
0.00 39.70 0.00 3.50 0.00 0.00 3.50 0.00 0.00 -3.50 0.00 0.00
0.00 39.80 0.00 4.00 0.00 0.00 4.00 0.00 0.00 -4.00 0.00 0.00
0.00 39.90 0.00 4.50 0.00 0.00 4.50 0.00 0.00 -4.50 0.00 0.00
"""

def create_test_files():
    """Create test BVH files for seed, build, and blend categories"""
    
    # Create test directories
    os.makedirs("test_data/seed", exist_ok=True)
    os.makedirs("test_data/build", exist_ok=True)
    os.makedirs("test_data/blend", exist_ok=True)
    
    # Create 2 seed files
    for i in range(1, 3):
        with open(f"test_data/seed/walk_{i:02d}.bvh", "w") as f:
            f.write(MINIMAL_BVH)
        print(f"✅ Created test_data/seed/walk_{i:02d}.bvh")
    
    # Create 2 build files
    for i in range(1, 3):
        with open(f"test_data/build/build_{i:02d}.bvh", "w") as f:
            f.write(MINIMAL_BVH)
        print(f"✅ Created test_data/build/build_{i:02d}.bvh")
    
    # Create 2 blend files
    for i in range(1, 3):
        with open(f"test_data/blend/blend_{i:02d}.bvh", "w") as f:
            f.write(MINIMAL_BVH)
        print(f"✅ Created test_data/blend/blend_{i:02d}.bvh")
    
    print("\n✅ Created 6 test BVH files (2 per category)")
    print("\nNext steps:")
    print("1. Create GCS bucket: gsutil mb gs://motionblend-mocap")
    print("2. Upload files: gsutil -m cp -r test_data/* gs://motionblend-mocap/mocap/")
    print("3. Run sync: make sync-test")

if __name__ == "__main__":
    create_test_files()
