#!/usr/bin/env python3
"""
Test bulk indexing with Elasticsearch Cloud
"""

from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime

# Elasticsearch Cloud configuration
client = Elasticsearch(
    "https://my-elasticsearch-project-ba986d.es.us-central1.gcp.elastic.cloud:443",
    api_key="S21qNXlKa0JEeUlTSnowSHBZRWg6VlVXWTd4Q0JPbDRSMC1KajFLQ2hKZw==",
)

index_name = "motion-blend"

# Test documents with proper motion capture field structure
docs = [
    {
        "_index": index_name,
        "_source": {
            "id": "test_motion_001",
            "name": "Yellowstone Wildlife Motion",
            "description": "Yellowstone National Park is one of the largest national parks in the United States. It ranges from the Wyoming to Montana and Idaho, and contains an area of 2,219,791 acres across three different states. Its most famous for hosting the geyser Old Faithful and is centered on the Yellowstone Caldera, the largest super volcano on the American continent.",
            "motion_vector": [9.142, 4.128, 6.436, 4.139, 4.933, 7.072, 7.425, 1.571],
            "metadata": {
                "frames": 150,
                "joints": 24,
                "duration": 5.0,
                "format": "FBX",
                "category": "nature",
                "tags": ["wildlife", "park", "animals"],
                "fps": 30.0,
                "file_size": 2048576
            },
            "created_at": "2025-10-09T12:41:01.024Z",
            "updated_at": "2025-10-09T12:41:01.113Z",
            "quality_score": 82.17,
            "motion_type": "environmental",
            "complexity": 69.77,
            "blend_compatibility": "high",
            "file_path": "/motions/yellowstone_wildlife.fbx",
            "checksum": "sha256_yellowstone_hash"
        }
    },
    {
        "_index": index_name,
        "_source": {
            "id": "test_motion_002", 
            "name": "Yosemite Climbing Motion",
            "description": "Yosemite National Park is a United States National Park, covering over 750,000 acres of land in California. A UNESCO World Heritage Site, the park is best known for its granite cliffs, waterfalls and giant sequoia trees. The park is home to a diverse range of wildlife and is a popular destination for rock climbers.",
            "motion_vector": [4.769, 0.295, 5.171, 5.448, 4.439, 2.169, 0.265, 6.543],
            "metadata": {
                "frames": 200,
                "joints": 24,
                "duration": 6.67,
                "format": "GLB",
                "category": "sports",
                "tags": ["climbing", "athletic", "granite"],
                "fps": 30.0,
                "file_size": 3145728
            },
            "created_at": "2025-10-09T12:41:01.113Z",
            "updated_at": "2025-10-09T12:41:01.113Z",
            "quality_score": 86.56,
            "motion_type": "athletic",
            "complexity": 56.47,
            "blend_compatibility": "medium",
            "file_path": "/motions/yosemite_climbing.glb",
            "checksum": "sha256_yosemite_hash"
        }
    },
    {
        "_index": index_name,
        "_source": {
            "id": "test_motion_003",
            "name": "Rocky Mountain Hiking Motion", 
            "description": "Rocky Mountain National Park is one of the most popular national parks in the United States. It receives over 4.5 million visitors annually, and is known for its mountainous terrain, including Longs Peak, which is the highest peak in the park.",
            "motion_vector": [2.521, 4.058, 4.146, 8.296, 0.521, 7.19, 1.44, 4.033],
            "metadata": {
                "frames": 120,
                "joints": 24,
                "duration": 4.0,
                "format": "TRC",
                "category": "locomotion",
                "tags": ["hiking", "mountain", "outdoor"],
                "fps": 30.0,
                "file_size": 1572864
            },
            "created_at": "2025-10-09T12:41:01.113Z",
            "updated_at": "2025-10-09T12:41:01.113Z", 
            "quality_score": 37.53,
            "motion_type": "locomotion",
            "complexity": 38.97,
            "blend_compatibility": "high",
            "file_path": "/motions/rocky_mountain_hiking.trc",
            "checksum": "sha256_rocky_hash"
        }
    }
]

def test_connection():
    """Test Elasticsearch connection"""
    print("🔗 Testing Elasticsearch Connection")
    print("=" * 50)
    
    try:
        info = client.info()
        print(f"✅ Connected to Elasticsearch {info['version']['number']}")
        print(f"   Cluster: {info['cluster_name']}")
        print(f"   Node: {info['name']}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def check_index():
    """Check if index exists and show mapping"""
    print(f"\n📋 Checking Index: {index_name}")
    print("=" * 50)
    
    try:
        if client.indices.exists(index=index_name):
            print(f"✅ Index '{index_name}' exists")
            
            # Get mapping
            mapping = client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings']['properties']
            
            print(f"   Fields: {len(properties)}")
            for field_name, field_config in list(properties.items())[:5]:
                field_type = field_config.get('type', 'object')
                print(f"   • {field_name}: {field_type}")
            
            if len(properties) > 5:
                print(f"   ... and {len(properties) - 5} more fields")
                
        else:
            print(f"⚠️  Index '{index_name}' does not exist")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error checking index: {e}")
        return False

def bulk_index_documents():
    """Perform bulk indexing"""
    print(f"\n📤 Bulk Indexing Documents")
    print("=" * 50)
    
    try:
        print(f"Preparing {len(docs)} documents for indexing...")
        
        # Timeout to allow machine learning model loading and semantic ingestion to complete
        ingestion_timeout = 300
        
        print(f"Starting bulk index (timeout: {ingestion_timeout}s)...")
        bulk_response = helpers.bulk(
            client.options(request_timeout=ingestion_timeout),
            docs,
            index=index_name
        )
        
        print(f"✅ Bulk indexing completed!")
        print(f"   Indexed: {bulk_response[0]} documents")
        print(f"   Errors: {len(bulk_response[1]) if bulk_response[1] else 0}")
        
        if bulk_response[1]:
            print("   Error details:")
            for error in bulk_response[1][:3]:  # Show first 3 errors
                print(f"   • {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bulk indexing failed: {e}")
        return False

def test_search():
    """Test search after indexing"""
    print(f"\n🔍 Testing Search After Indexing")
    print("=" * 50)
    
    try:
        # Wait for documents to be available
        client.indices.refresh(index=index_name)
        
        # Test vector search
        vector_query = {
            "size": 3,
            "query": {
                "knn": {
                    "field": "motion_vector",
                    "query_vector": [5.0, 3.0, 4.0, 6.0, 2.0, 5.0, 3.0, 4.0],
                    "k": 3,
                    "num_candidates": 10
                }
            }
        }
        
        print("Vector search results:")
        response = client.search(index=index_name, body=vector_query)
        for i, hit in enumerate(response['hits']['hits'], 1):
            score = hit['_score']
            name = hit['_source']['name'][:50] + "..." if len(hit['_source']['name']) > 50 else hit['_source']['name']
            print(f"   {i}. {name} (score: {score:.3f})")
        
        # Test text search 
        text_query = {
            "size": 3,
            "query": {
                "multi_match": {
                    "query": "mountain climbing",
                    "fields": ["name^2", "description", "metadata.tags^3"]
                }
            }
        }
        
        print("\nText search results:")
        response = client.search(index=index_name, body=text_query)
        for i, hit in enumerate(response['hits']['hits'], 1):
            score = hit['_score']
            name = hit['_source']['name'][:50] + "..." if len(hit['_source']['name']) > 50 else hit['_source']['name']
            print(f"   {i}. {name} (score: {score:.3f})")
            
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 Elasticsearch Cloud Bulk Indexing Test")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Connection Test", test_connection),
        ("Index Check", check_index),
        ("Bulk Indexing", bulk_index_documents),
        ("Search Test", test_search)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"\n❌ {test_name} failed - stopping execution")
            break
    
    print(f"\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests completed successfully")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Elasticsearch Cloud integration is working!")
        print("\n📝 Next steps:")
        print("1. Update Flask app to use bulk indexing endpoint")
        print("2. Implement real motion capture data ingestion")
        print("3. Add semantic search with ELSER model")
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed")

if __name__ == "__main__":
    main()