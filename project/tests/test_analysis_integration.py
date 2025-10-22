"""
Test Analysis Integration
=========================

Tests for motion blend analysis integration with the artifact system.
Verifies that analysis data is correctly generated, stored, and retrieved.
"""

import pytest
import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "elastic_search"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Flask app
from app import app, ARTIFACTS_STORE, generate_blend_analysis


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_blend_artifact():
    """Create a sample blend artifact for testing"""
    return {
        "id": "test_blend_001",
        "name": "Walking_Running_blend_0.50",
        "type": "motion_blend",
        "metadata": {
            "source_motions": ["Walking", "Running"],
            "blend_weight": 0.5,
            "frames": 120,
            "duration": 4.0,
            "motion_hash": 12345
        }
    }


class TestBlendAnalysisGeneration:
    """Test analysis data generation"""
    
    def test_generate_blend_analysis_structure(self):
        """Test that generated analysis has correct structure"""
        analysis = generate_blend_analysis(
            motion_hash=12345,
            frames=120,
            duration=4.0,
            weight=0.5
        )
        
        # Check top-level keys
        assert "l2_velocity" in analysis
        assert "l2_acceleration" in analysis
        assert "metrics" in analysis
        assert "transition_window" in analysis
        assert "joint_names" in analysis
        assert "time_points" in analysis
        
    def test_generate_blend_analysis_joints(self):
        """Test that all required joints are present"""
        analysis = generate_blend_analysis(
            motion_hash=12345,
            frames=120,
            duration=4.0,
            weight=0.5
        )
        
        expected_joints = ['Hips', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot']
        
        for joint in expected_joints:
            assert joint in analysis["l2_velocity"]
            assert joint in analysis["l2_acceleration"]
            assert len(analysis["l2_velocity"][joint]) == 120
            assert len(analysis["l2_acceleration"][joint]) == 120
    
    def test_generate_blend_analysis_metrics(self):
        """Test that metrics are computed correctly"""
        analysis = generate_blend_analysis(
            motion_hash=12345,
            frames=120,
            duration=4.0,
            weight=0.5
        )
        
        metrics = analysis["metrics"]
        
        # Check all metrics exist
        assert "mean_velocity" in metrics
        assert "std_velocity" in metrics
        assert "max_velocity" in metrics
        assert "mean_acceleration" in metrics
        assert "std_acceleration" in metrics
        assert "transition_smoothness" in metrics
        assert "global_diversity" in metrics
        assert "quality_score" in metrics
        assert "quality_category" in metrics
        
        # Check value ranges
        assert 0 <= metrics["quality_score"] <= 1
        assert metrics["quality_category"] in ["good", "fair", "poor"]
        assert metrics["mean_velocity"] >= 0
        assert metrics["mean_acceleration"] >= 0
    
    def test_generate_blend_analysis_deterministic(self):
        """Test that analysis is deterministic with same seed"""
        analysis1 = generate_blend_analysis(12345, 120, 4.0, 0.5)
        analysis2 = generate_blend_analysis(12345, 120, 4.0, 0.5)
        
        # Should generate identical data with same seed
        assert analysis1["l2_velocity"]["Hips"][0] == analysis2["l2_velocity"]["Hips"][0]
        assert analysis1["metrics"]["mean_velocity"] == analysis2["metrics"]["mean_velocity"]
    
    def test_generate_blend_analysis_unique(self):
        """Test that different seeds produce different analysis"""
        analysis1 = generate_blend_analysis(12345, 120, 4.0, 0.5)
        analysis2 = generate_blend_analysis(54321, 120, 4.0, 0.5)
        
        # Should generate different data with different seeds
        assert analysis1["l2_velocity"]["Hips"][50] != analysis2["l2_velocity"]["Hips"][50]


class TestBlendCreationAPI:
    """Test blend creation API with analysis"""
    
    def test_create_blend_includes_analysis(self, client):
        """Test that creating a blend includes analysis data"""
        response = client.post('/api/blend', json={
            "motion1": "Walking",
            "motion2": "Running",
            "weight": 0.5
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data["status"] == "success"
        assert "artifact" in data
        
        artifact = data["artifact"]
        assert "analysis" in artifact
        assert "l2_velocity" in artifact["analysis"]
        assert "metrics" in artifact["analysis"]
        assert "quality_score" in artifact["metadata"]
        assert "quality_category" in artifact["metadata"]
    
    def test_create_blend_quality_metadata(self, client):
        """Test that quality metrics are in artifact metadata"""
        response = client.post('/api/blend', json={
            "motion1": "Walking",
            "motion2": "Running",
            "weight": 0.7
        })
        
        data = json.loads(response.data)
        artifact = data["artifact"]
        
        assert "quality_score" in artifact["metadata"]
        assert "quality_category" in artifact["metadata"]
        assert 0 <= artifact["metadata"]["quality_score"] <= 1
        assert artifact["metadata"]["quality_category"] in ["good", "fair", "poor"]


class TestAnalysisRetrievalAPI:
    """Test analysis retrieval endpoints"""
    
    def test_get_artifact_analysis_existing(self, client, sample_blend_artifact):
        """Test retrieving analysis for existing artifact"""
        # Add artifact to store
        ARTIFACTS_STORE.clear()
        analysis_data = generate_blend_analysis(12345, 120, 4.0, 0.5)
        sample_blend_artifact["analysis"] = analysis_data
        sample_blend_artifact["metadata"]["quality_score"] = analysis_data["metrics"]["quality_score"]
        sample_blend_artifact["metadata"]["quality_category"] = analysis_data["metrics"]["quality_category"]
        ARTIFACTS_STORE.append(sample_blend_artifact)
        
        response = client.get('/api/artifact/test_blend_001/analysis')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "analysis" in data
        assert "artifact_id" in data
        assert "artifact_name" in data
        assert data["artifact_id"] == "test_blend_001"
    
    def test_get_artifact_analysis_generates_on_demand(self, client, sample_blend_artifact):
        """Test that analysis is generated on-demand if not present"""
        # Add artifact without analysis
        ARTIFACTS_STORE.clear()
        ARTIFACTS_STORE.append(sample_blend_artifact)
        
        response = client.get('/api/artifact/test_blend_001/analysis')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "analysis" in data
        assert "generated" in data
        assert data["generated"] == "on-demand"
    
    def test_get_artifact_analysis_not_found(self, client):
        """Test error handling for non-existent artifact"""
        ARTIFACTS_STORE.clear()
        
        response = client.get('/api/artifact/nonexistent/analysis')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


class TestArtifactsManifest:
    """Test artifacts manifest includes analysis metadata"""
    
    def test_manifest_includes_quality_info(self, client):
        """Test that manifest shows quality information"""
        # Create a blend with analysis
        client.post('/api/blend', json={
            "motion1": "Walking",
            "motion2": "Running",
            "weight": 0.5
        })
        
        response = client.get('/api/artifacts/manifest')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "artifacts" in data
        assert len(data["artifacts"]) > 0
        
        # Check first artifact has quality metadata
        artifact = data["artifacts"][0]
        if artifact.get("type") == "motion_blend":
            assert "quality_score" in artifact.get("metadata", {})
            assert "quality_category" in artifact.get("metadata", {})


class TestAnalysisDataFormat:
    """Test analysis data format compatibility"""
    
    def test_analysis_compatible_with_frontend(self):
        """Test that analysis format matches frontend expectations"""
        analysis = generate_blend_analysis(12345, 120, 4.0, 0.5)
        
        # Check joint names match what frontend expects
        # Frontend expects: Hips (maps to Pelvis), LeftWrist, RightWrist, LeftFoot, RightFoot
        assert "Hips" in analysis["l2_velocity"]  # Backend uses "Hips"
        assert "LeftWrist" in analysis["l2_velocity"]
        assert "RightWrist" in analysis["l2_velocity"]
        assert "LeftFoot" in analysis["l2_velocity"]
        assert "RightFoot" in analysis["l2_velocity"]
        
        # Check transition window format
        assert "start" in analysis["transition_window"]
        assert "end" in analysis["transition_window"]
        
        # Check metrics format
        metrics = analysis["metrics"]
        assert all(isinstance(v, (int, float)) for k, v in metrics.items() if k != "quality_category")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
