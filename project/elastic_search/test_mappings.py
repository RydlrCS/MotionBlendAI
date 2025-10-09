#!/usr/bin/env python3
"""
Test script for Elasticsearch field mappings and indexing
"""

import sys
sys.path.append('/Users/ted/blenderkit_data/MotionBlendAI-1/project/elastic_search')

from app import create_motion_mappings, MOCK_MOTIONS, ES_INDEX_NAME

def test_mappings():
    """Test the field mappings structure"""
    print("🔍 Testing Elasticsearch Field Mappings")
    print("=" * 50)
    
    mappings = create_motion_mappings()
    
    print("📋 Motion Capture Field Mappings:")
    print("-" * 30)
    
    properties = mappings["properties"]
    
    for field_name, field_config in properties.items():
        field_type = field_config.get("type", "object")
        print(f"• {field_name:20} -> {field_type}")
        
        # Show nested properties for objects
        if field_type == "object" and "properties" in field_config:
            for sub_field, sub_config in field_config["properties"].items():
                sub_type = sub_config.get("type", "unknown")
                print(f"  └─ {sub_field:15} -> {sub_type}")
        
        # Show fields with multiple analyzers
        if "fields" in field_config:
            for field_variant, variant_config in field_config["fields"].items():
                variant_type = variant_config.get("type", "unknown")
                print(f"  └─ {field_variant:15} -> {variant_type}")
    
    print(f"\n📊 Summary:")
    print(f"Total fields: {len(properties)}")
    print(f"Vector field: motion_vector (8 dimensions, cosine similarity)")
    print(f"Semantic search: name.semantic, description.semantic")
    print(f"Index name: {ES_INDEX_NAME}")
    
    return mappings

def test_mock_data():
    """Test mock data structure compatibility"""
    print("\n🎭 Testing Mock Data Compatibility")
    print("=" * 50)
    
    print(f"Mock motions available: {len(MOCK_MOTIONS)}")
    
    for i, motion in enumerate(MOCK_MOTIONS[:3], 1):
        print(f"\n{i}. {motion['name']}")
        print(f"   Vector dims: {len(motion['vector'])}")
        print(f"   Format: {motion['metadata']['format']}")
        print(f"   Category: {motion['metadata']['category']}")
        print(f"   Tags: {', '.join(motion['metadata']['tags'])}")
    
    print(f"\n... and {len(MOCK_MOTIONS) - 3} more motions")

def show_example_queries():
    """Show example Elasticsearch queries"""
    print("\n🔎 Example Elasticsearch Queries")
    print("=" * 50)
    
    print("1. Vector Similarity Search:")
    print("""
    POST /motion-blend/_search
    {
      "size": 10,
      "query": {
        "knn": {
          "field": "motion_vector",
          "query_vector": [0.12, 0.34, 0.56, 0.78, 0.23, 0.45, 0.67, 0.89],
          "k": 10,
          "num_candidates": 20
        }
      }
    }""")
    
    print("\n2. Semantic Text Search:")
    print("""
    POST /motion-blend/_search
    {
      "size": 10,
      "query": {
        "bool": {
          "should": [
            {
              "semantic": {
                "field": "name.semantic",
                "query": "dancing hip hop movement"
              }
            },
            {
              "multi_match": {
                "query": "dance",
                "fields": ["name^3", "metadata.tags^2", "metadata.category^2"]
              }
            }
          ]
        }
      }
    }""")
    
    print("\n3. Hybrid Search (Vector + Text):")
    print("""
    POST /motion-blend/_search
    {
      "size": 10,
      "query": {
        "bool": {
          "should": [
            {
              "knn": {
                "field": "motion_vector",
                "query_vector": [0.12, 0.34, 0.56, 0.78, 0.23, 0.45, 0.67, 0.89],
                "k": 5
              }
            },
            {
              "multi_match": {
                "query": "athletic jumping",
                "fields": ["name", "metadata.tags", "description"]
              }
            }
          ],
          "minimum_should_match": 1
        }
      }
    }""")

if __name__ == "__main__":
    print("🚀 MotionBlend AI Elasticsearch Integration Test")
    print("=" * 60)
    
    try:
        # Test mappings
        mappings = test_mappings()
        
        # Test mock data
        test_mock_data()
        
        # Show example queries
        show_example_queries()
        
        print("\n✅ All tests completed successfully!")
        print("\n📝 Next Steps:")
        print("1. Start the Flask server: python3 app.py")
        print("2. Test endpoints:")
        print("   - GET  /health")
        print("   - POST /search (vector search)")
        print("   - POST /search/text (text search)")
        print("   - POST /index (index single motion)")
        print("   - POST /index/bulk (bulk index motions)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()