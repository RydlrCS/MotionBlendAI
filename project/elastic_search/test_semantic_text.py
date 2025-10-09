#!/usr/bin/env python3
"""
Test script for Elasticsearch semantic text functionality
Using the new cluster configuration and semantic_text field type
"""

from elasticsearch import Elasticsearch, helpers
import json

def test_semantic_text_setup():
    """Test semantic text setup and document ingestion"""
    print("🔍 Testing Elasticsearch Semantic Text Setup")
    print("=" * 60)
    
    # Connect to the new Elasticsearch cluster
    client = Elasticsearch(
        "https://my-elasticsearch-project-bb39cc.es.us-central1.gcp.elastic.cloud:443",
        api_key="bHRLcXlaa0JSaHFSM2NuRk9tYVA6QUY3NXE2SkQzaWdYWWkxelZLLXQxZw=="
    )
    
    index_name = "motion-blend"
    
    # Test connection
    try:
        info = client.info()
        print(f"✅ Connected to Elasticsearch {info['version']['number']}")
        print(f"   Cluster: {info['cluster_name']}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Create semantic text mapping
    print(f"\n📋 Creating semantic text mapping for index: {index_name}")
    mappings = {
        "properties": {
            "text": {
                "type": "semantic_text"
            }
        }
    }
    
    try:
        mapping_response = client.indices.put_mapping(index=index_name, body=mappings)
        print(f"✅ Mapping created successfully: {mapping_response}")
    except Exception as e:
        print(f"⚠️ Mapping update result: {e}")
        # Continue anyway, mapping might already exist
    
    # Prepare documents for bulk ingestion
    print(f"\n📤 Preparing documents for bulk ingestion...")
    docs = [
        {
            "_index": index_name,
            "_source": {
                "text": "Yellowstone National Park is one of the largest national parks in the United States. It ranges from the Wyoming to Montana and Idaho, and contains an area of 2,219,791 acres across three different states. Its most famous for hosting the geyser Old Faithful and is centered on the Yellowstone Caldera, the largest super volcano on the American continent. Yellowstone is host to hundreds of species of animal, many of which are endangered or threatened. Most notably, it contains free-ranging herds of bison and elk, alongside bears, cougars and wolves. The national park receives over 4.5 million visitors annually and is a UNESCO World Heritage Site."
            }
        },
        {
            "_index": index_name,
            "_source": {
                "text": "Yosemite National Park is a United States National Park, covering over 750,000 acres of land in California. A UNESCO World Heritage Site, the park is best known for its granite cliffs, waterfalls and giant sequoia trees. Yosemite hosts over four million visitors in most years, with a peak of five million visitors in 2016. The park is home to a diverse range of wildlife, including mule deer, black bears, and the endangered Sierra Nevada bighorn sheep. The park has 1,200 square miles of wilderness, and is a popular destination for rock climbers, with over 3,000 feet of vertical granite to climb. Its most famous cliff is the El Capitan, a 3,000 feet monolith along its tallest face."
            }
        },
        {
            "_index": index_name,
            "_source": {
                "text": "Rocky Mountain National Park is one of the most popular national parks in the United States. It receives over 4.5 million visitors annually, and is known for its mountainous terrain, including Longs Peak, which is the highest peak in the park. The park is home to a variety of wildlife, including elk, mule deer, moose, and bighorn sheep. The park is also home to a variety of ecosystems, including montane, subalpine, and alpine tundra. The park is a popular destination for hiking, camping, and wildlife viewing, and is a UNESCO World Heritage Site."
            }
        }
    ]
    
    # Perform bulk ingestion with extended timeout for ML model loading
    print(f"📦 Starting bulk ingestion with {len(docs)} documents...")
    print("⏳ Using 300s timeout to allow for machine learning model loading...")
    
    try:
        ingestion_timeout = 300
        bulk_response = helpers.bulk(
            client.options(request_timeout=ingestion_timeout),
            docs,
            index=index_name
        )
        
        print(f"✅ Bulk ingestion completed!")
        print(f"   Successful: {bulk_response[0]} documents")
        print(f"   Errors: {len(bulk_response[1]) if bulk_response[1] else 0}")
        
        if bulk_response[1]:
            print("   Error details:")
            for error in bulk_response[1][:3]:
                print(f"   • {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bulk ingestion failed: {e}")
        return False

def test_semantic_search():
    """Test semantic search functionality"""
    print(f"\n🔍 Testing Semantic Search")
    print("=" * 40)
    
    client = Elasticsearch(
        "https://my-elasticsearch-project-bb39cc.es.us-central1.gcp.elastic.cloud:443",
        api_key="bHRLcXlaa0JSaHFSM2NuRk9tYVA6QUY3NXE2SkQzaWdYWWkxelZLLXQxZw=="
    )
    
    index_name = "motion-blend"
    
    try:
        # Wait for documents to be indexed
        client.indices.refresh(index=index_name)
        
        # Test semantic search query
        search_query = {
            "size": 5,
            "query": {
                "semantic": {
                    "field": "text",
                    "query": "wildlife and mountains with animals"
                }
            }
        }
        
        print("🔎 Searching for: 'wildlife and mountains with animals'")
        response = client.search(index=index_name, body=search_query)
        
        print(f"📊 Found {response['hits']['total']['value']} results:")
        for i, hit in enumerate(response['hits']['hits'], 1):
            score = hit['_score']
            text_preview = hit['_source']['text'][:100] + "..." if len(hit['_source']['text']) > 100 else hit['_source']['text']
            print(f"   {i}. Score: {score:.3f}")
            print(f"      Text: {text_preview}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 Elasticsearch Semantic Text Integration Test")
    print("=" * 70)
    print("🌐 New Cluster: my-elasticsearch-project-bb39cc")
    print("🔑 API Key: bHRLcXlaa0JSaHFSM2NuRk9tYVA6...")
    print()
    
    tests = [
        ("Semantic Text Setup", test_semantic_text_setup),
        ("Semantic Search", test_semantic_search)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"\n❌ {test_name} failed - continuing with remaining tests")
    
    print(f"\n📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests completed successfully")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Semantic text functionality is working!")
        print("\n📝 Next steps:")
        print("1. Update Flask app with new credentials")
        print("2. Integrate semantic search into API endpoints")
        print("3. Test motion capture data with semantic fields")
    else:
        print(f"\n⚠️ {len(tests) - passed} test(s) failed")

if __name__ == "__main__":
    main()