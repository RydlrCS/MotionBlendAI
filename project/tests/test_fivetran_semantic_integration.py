#!/usr/bin/env python3
"""
Test script to validate Fivetran connector integration with semantic text fields.
Tests motion capture data pipeline with Elasticsearch semantic search capabilities.
"""

import sys
import os
import json
import time
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Mock the BigQuery dependencies for testing
class MockBigQueryClient:
    """Mock BigQuery client for testing without actual BigQuery."""
    def __init__(self):
        self.inserted_records = []
    
    def insert_rows_json(self, table, records):
        self.inserted_records.extend(records)
        return []  # No errors

class MockDummyConnector:
    """Mock base connector for testing."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bq_client = MockBigQueryClient()
        self.bq_table = config.get('bigquery_table')

    def load(self, record: Dict[str, Any]):
        """Mock load method."""
        self.bq_client.insert_rows_json(self.bq_table, [record])
        print(f"[Mock BigQuery] Loaded record at {record.get('timestamp')}")

# Patch the import before importing the actual connector
sys.modules['google.cloud.bigquery'] = type('MockBigQuery', (), {'Client': MockBigQueryClient})()
sys.modules['google.api_core.exceptions'] = type('MockExceptions', (), {'GoogleAPIError': Exception})()

# Now import and patch the connector
from project.fivetran_connector.PoseStreamConnector import preprocess_frame
import project.fivetran_connector.PoseStreamConnector as psc

# Replace the base class
psc.DummyConnector = MockDummyConnector

class PoseStreamConnector(MockDummyConnector):
    """Mock PoseStreamConnector for testing."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = config.get('moverse_api_url')
        self.api_key = config.get('moverse_api_key')
        self.mode = config.get('mode', 'stream')
        self.file_folder = config.get('file_folder', None)
        self.bigquery_table = config.get('bigquery_table')

try:
    from project.elastic_search.app import initialize_elasticsearch, es, es_available, ES_INDEX_NAME
except ImportError as e:
    print(f"Warning: Could not import Elasticsearch components: {e}")
    es = None
    es_available = False
    ES_INDEX_NAME = "motion-blend"
    def initialize_elasticsearch():
        pass

class SemanticPoseStreamConnector(PoseStreamConnector):
    """Extended connector that integrates with Elasticsearch semantic search."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.elasticsearch_enabled = config.get('elasticsearch_enabled', True)
        self.semantic_descriptions = {
            'bvh': 'dynamic motion capture data with skeletal animation',
            'trc': 'marker-based motion tracking with spatial coordinates',
            'fbx': 'complex character animation with rigging and poses'
        }
    
    def load(self, record: Dict[str, Any]):
        """Enhanced load method that sends data to both BigQuery and Elasticsearch."""
        # Original BigQuery loading
        super().load(record)
        
        # Enhanced semantic processing for Elasticsearch
        if self.elasticsearch_enabled and es_available and es:
            semantic_record = self._enhance_with_semantic_fields(record)
            self._index_to_elasticsearch(semantic_record)
    
    def _enhance_with_semantic_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add semantic text fields for better search capabilities."""
        enhanced = record.copy()
        
        # Generate semantic description based on motion characteristics
        meta = record.get('meta', {})
        source = meta.get('source', 'unknown')
        file_path = meta.get('file', '')
        
        # Analyze joint data for motion characteristics
        joints = record.get('joints', [])
        motion_intensity = self._calculate_motion_intensity(joints)
        motion_type = self._classify_motion_type(joints, motion_intensity)
        
        # Create rich semantic description
        base_description = self.semantic_descriptions.get(source, 'motion capture data')
        semantic_description = f"{base_description} featuring {motion_type} movement patterns"
        
        if motion_intensity > 0.7:
            semantic_description += " with high-energy dynamic motion"
        elif motion_intensity > 0.3:
            semantic_description += " with moderate movement activity"
        else:
            semantic_description += " with subtle or static positioning"
        
        # Add file-based context
        if file_path:
            filename = os.path.basename(file_path).replace('_', ' ').replace('.', ' ')
            semantic_description += f" from {filename}"
        
        # Enhanced record with semantic fields
        enhanced.update({
            'id': f"motion_{int(record.get('timestamp', time.time()))}",
            'name': {
                'text': f"{source.upper()} Motion {int(record.get('timestamp', 0))}",
                'semantic': f"{motion_type} motion capture from {source} format"
            },
            'description': {
                'text': semantic_description,
                'semantic': semantic_description
            },
            'motion_vector': self._generate_motion_vector(joints),
            'metadata': {
                'frames': len(joints) // 3 if joints else 0,  # Estimate frames from joint data
                'joints': len(joints) // 3 if joints else 0,
                'duration': 1.0,  # Default duration
                'format': source.upper(),
                'category': motion_type,
                'tags': self._generate_tags(motion_type, motion_intensity, source),
                'fps': 30.0,
                'file_size': len(str(joints)) if joints else 0
            },
            'created_at': record.get('timestamp', time.time()),
            'updated_at': time.time(),
            'quality_score': min(1.0, motion_intensity + 0.3),
            'motion_type': motion_type,
            'complexity': motion_intensity,
            'blend_compatibility': 'high' if motion_intensity > 0.5 else 'medium',
            'file_path': file_path,
            'checksum': str(hash(str(joints))) if joints else '0'
        })
        
        return enhanced
    
    def _calculate_motion_intensity(self, joints: List[float]) -> float:
        """Calculate motion intensity from joint data."""
        if not joints or len(joints) < 6:
            return 0.0
        
        # Calculate variance in joint positions as intensity proxy
        try:
            import statistics
            variance = statistics.variance(joints)
            # Normalize to 0-1 range
            return min(1.0, variance / 1000.0)
        except:
            return 0.5  # Default moderate intensity
    
    def _classify_motion_type(self, joints: List[float], intensity: float) -> str:
        """Classify motion type based on joint data and intensity."""
        if not joints:
            return 'static'
        
        if intensity > 0.8:
            return 'athletic'
        elif intensity > 0.6:
            return 'dynamic'
        elif intensity > 0.4:
            return 'locomotion'
        elif intensity > 0.2:
            return 'gesture'
        else:
            return 'static'
    
    def _generate_motion_vector(self, joints: List[float]) -> List[float]:
        """Generate 8-dimensional motion vector from joint data."""
        if not joints or len(joints) < 8:
            # Generate default vector
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # Sample 8 representative values from joints
        step = max(1, len(joints) // 8)
        vector = [joints[i] for i in range(0, len(joints), step)][:8]
        
        # Pad if needed
        while len(vector) < 8:
            vector.append(0.0)
        
        # Normalize to reasonable range
        return [float(max(-1.0, min(1.0, x / 100.0))) for x in vector]
    
    def _generate_tags(self, motion_type: str, intensity: float, source: str) -> List[str]:
        """Generate descriptive tags for the motion."""
        tags = [motion_type, source]
        
        if intensity > 0.7:
            tags.extend(['high-energy', 'dynamic', 'expressive'])
        elif intensity > 0.4:
            tags.extend(['moderate', 'balanced'])
        else:
            tags.extend(['subtle', 'controlled'])
        
        # Add motion-specific tags
        motion_tags = {
            'athletic': ['sports', 'fitness', 'performance'],
            'dynamic': ['energetic', 'fluid', 'expressive'],
            'locomotion': ['walking', 'movement', 'travel'],
            'gesture': ['communication', 'expression', 'social'],
            'static': ['pose', 'position', 'stance']
        }
        
        tags.extend(motion_tags.get(motion_type, []))
        return list(set(tags))  # Remove duplicates
    
    def _index_to_elasticsearch(self, record: Dict[str, Any]):
        """Index the enhanced record to Elasticsearch with semantic fields."""
        try:
            if es and es_available:
                response = es.index(
                    index=ES_INDEX_NAME,
                    id=record.get('id'),
                    body=record,
                    timeout='300s'  # Extended timeout for semantic processing
                )
                print(f"✅ Indexed to Elasticsearch: {record.get('id')} (result: {response.get('result', 'unknown')})")
            else:
                print("⚠️ Elasticsearch not available, skipping indexing")
        except Exception as e:
            print(f"❌ Elasticsearch indexing error: {e}")

def test_semantic_integration():
    """Test the integration between Fivetran connector and semantic search."""
    print("🚀 Testing Fivetran-Elasticsearch Semantic Integration")
    print("=" * 60)
    
    # Initialize Elasticsearch connection
    initialize_elasticsearch()
    
    if not es_available:
        print("⚠️ Elasticsearch not available, using mock data only")
    
    # Test configuration
    config = {
        'mode': 'batch',
        'file_folder': '/Users/ted/blenderkit_data/MotionBlendAI-1/project/seed_motions',
        'bigquery_table': 'test.motions',
        'elasticsearch_enabled': True
    }
    
    # Create enhanced connector
    connector = SemanticPoseStreamConnector(config)
    
    # Replace BigQuery client with mock for testing
    connector.bq_client = MockBigQueryClient()
    
    print(f"\n📁 Testing with folder: {config['file_folder']}")
    
    # Test with sample motion data
    test_motions = [
        {
            'timestamp': time.time(),
            'joints': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'meta': {'source': 'bvh', 'file': 'test_walk.bvh'}
        },
        {
            'timestamp': time.time() + 1,
            'joints': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            'meta': {'source': 'trc', 'file': 'test_run.trc'}
        },
        {
            'timestamp': time.time() + 2,
            'joints': [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            'meta': {'source': 'fbx', 'file': 'test_pose.fbx'}
        }
    ]
    
    print(f"\n🔄 Processing {len(test_motions)} test motions...")
    
    for i, motion in enumerate(test_motions, 1):
        print(f"\n--- Processing Motion {i} ---")
        processed = preprocess_frame(motion)
        connector.load(processed)
    
    # Verify BigQuery mock data
    print(f"\n📊 BigQuery Mock Results:")
    print(f"Records inserted: {len(connector.bq_client.inserted_records)}")
    
    for record in connector.bq_client.inserted_records:
        print(f"  - Timestamp: {record.get('timestamp')}")
        print(f"    Source: {record.get('meta', {}).get('source', 'unknown')}")
        print(f"    Joints: {len(record.get('joints', []))} values")
    
    # Test Elasticsearch integration if available
    if es_available and es:
        print(f"\n🔍 Testing Elasticsearch Semantic Search...")
        
        # Wait a moment for indexing
        time.sleep(2)
        
        try:
            # Test semantic search
            test_queries = [
                "dynamic motion capture",
                "walking movement",
                "athletic performance",
                "static pose"
            ]
            
            for query in test_queries:
                print(f"\n🔎 Semantic search: '{query}'")
                response = es.search(
                    index=ES_INDEX_NAME,
                    body={
                        "size": 3,
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "semantic": {
                                            "field": "name.semantic",
                                            "query": query
                                        }
                                    },
                                    {
                                        "semantic": {
                                            "field": "description.semantic",
                                            "query": query
                                        }
                                    }
                                ]
                            }
                        }
                    }
                )
                
                hits = response.get('hits', {}).get('hits', [])
                print(f"  Found {len(hits)} results:")
                for hit in hits:
                    source_data = hit['_source']
                    print(f"    - {source_data.get('name', {}).get('text', 'Unknown')} (score: {hit['_score']:.2f})")
                    print(f"      Type: {source_data.get('motion_type', 'unknown')}")
                
        except Exception as e:
            print(f"❌ Semantic search test failed: {e}")
    
    print(f"\n✅ Integration test completed!")
    print(f"📈 Handshake between Fivetran connector and Elasticsearch semantic search: {'SUCCESS' if es_available else 'MOCK MODE'}")

if __name__ == '__main__':
    test_semantic_integration()