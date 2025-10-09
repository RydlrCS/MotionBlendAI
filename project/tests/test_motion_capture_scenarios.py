#!/usr/bin/env python3
"""
Comprehensive test suite for motion capture specific semantic search scenarios.
Tests various motion categories, styles, and use cases with the semantic search API.
"""

import sys
import os
import json
import time
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, '/Users/ted/blenderkit_data/MotionBlendAI-1/project/elastic_search')

def create_motion_test_data() -> List[Dict[str, Any]]:
    """Create comprehensive motion capture test data covering various scenarios."""
    return [
        # Athletic motions
        {
            "id": "athletic_001",
            "name": "Professional Basketball Layup",
            "description": "Dynamic athletic basketball layup with explosive jumping and precise ball handling coordination",
            "motion_vector": [0.9, 0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.8],
            "metadata": {
                "category": "athletic",
                "tags": ["basketball", "jumping", "explosive", "sports", "coordination"],
                "duration": 3.2,
                "frames": 96,
                "fps": 30,
                "complexity": 0.85
            },
            "motion_type": "athletic",
            "quality_score": 0.92
        },
        {
            "id": "athletic_002", 
            "name": "Olympic Sprint Start",
            "description": "Professional sprinter explosive start from starting blocks with perfect form and timing",
            "motion_vector": [0.95, 0.9, 0.8, 0.85, 0.9, 0.7, 0.8, 0.9],
            "metadata": {
                "category": "athletic",
                "tags": ["sprint", "explosive", "olympic", "running", "start"],
                "duration": 2.1,
                "frames": 63,
                "fps": 30,
                "complexity": 0.9
            },
            "motion_type": "athletic",
            "quality_score": 0.95
        },
        
        # Dance motions
        {
            "id": "dance_001",
            "name": "Contemporary Dance Sequence",
            "description": "Flowing contemporary dance with graceful arm movements and expressive body language",
            "motion_vector": [0.6, 0.7, 0.8, 0.5, 0.6, 0.9, 0.7, 0.6],
            "metadata": {
                "category": "dance",
                "tags": ["contemporary", "graceful", "expressive", "flowing", "artistic"],
                "duration": 12.5,
                "frames": 375,
                "fps": 30,
                "complexity": 0.7
            },
            "motion_type": "dance",
            "quality_score": 0.88
        },
        {
            "id": "dance_002",
            "name": "Hip Hop Battle Freestyle",
            "description": "High-energy hip hop freestyle with sharp movements, freezes, and rhythmic footwork",
            "motion_vector": [0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.9, 0.7],
            "metadata": {
                "category": "dance",
                "tags": ["hip-hop", "freestyle", "rhythmic", "sharp", "urban", "battle"],
                "duration": 8.3,
                "frames": 249,
                "fps": 30,
                "complexity": 0.75
            },
            "motion_type": "dance",
            "quality_score": 0.82
        },
        
        # Martial arts motions
        {
            "id": "martial_001",
            "name": "Karate Kata Form",
            "description": "Traditional karate kata with precise strikes, blocks, and stances demonstrating martial discipline",
            "motion_vector": [0.7, 0.8, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7],
            "metadata": {
                "category": "martial-arts",
                "tags": ["karate", "kata", "traditional", "precise", "discipline", "strikes"],
                "duration": 15.8,
                "frames": 474,
                "fps": 30,
                "complexity": 0.8
            },
            "motion_type": "combat",
            "quality_score": 0.9
        },
        
        # Everyday motions
        {
            "id": "everyday_001",
            "name": "Professional Handshake",
            "description": "Confident business handshake with proper posture and professional demeanor",
            "motion_vector": [0.3, 0.4, 0.2, 0.3, 0.4, 0.3, 0.2, 0.4],
            "metadata": {
                "category": "gesture",
                "tags": ["handshake", "professional", "business", "greeting", "confident"],
                "duration": 1.8,
                "frames": 54,
                "fps": 30,
                "complexity": 0.3
            },
            "motion_type": "gesture",
            "quality_score": 0.75
        },
        {
            "id": "everyday_002",
            "name": "Casual Walking Stride",
            "description": "Natural everyday walking with relaxed pace and normal gait pattern",
            "motion_vector": [0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.5],
            "metadata": {
                "category": "locomotion",
                "tags": ["walking", "casual", "natural", "everyday", "relaxed"],
                "duration": 6.0,
                "frames": 180,
                "fps": 30,
                "complexity": 0.4
            },
            "motion_type": "locomotion",
            "quality_score": 0.7
        },
        
        # Performance motions
        {
            "id": "performance_001",
            "name": "Theater Dramatic Gesture",
            "description": "Theatrical dramatic gesture with exaggerated movements for stage performance",
            "motion_vector": [0.6, 0.8, 0.7, 0.5, 0.7, 0.8, 0.6, 0.7],
            "metadata": {
                "category": "performance",
                "tags": ["theater", "dramatic", "exaggerated", "stage", "acting"],
                "duration": 4.2,
                "frames": 126,
                "fps": 30,
                "complexity": 0.65
            },
            "motion_type": "gesture",
            "quality_score": 0.78
        },
        
        # Wellness/Yoga motions
        {
            "id": "wellness_001",
            "name": "Sunrise Yoga Flow",
            "description": "Peaceful morning yoga sequence with sun salutations and mindful breathing movements",
            "motion_vector": [0.3, 0.4, 0.5, 0.3, 0.4, 0.6, 0.4, 0.5],
            "metadata": {
                "category": "wellness",
                "tags": ["yoga", "sunrise", "peaceful", "mindful", "breathing", "flow"],
                "duration": 18.5,
                "frames": 555,
                "fps": 30,
                "complexity": 0.5
            },
            "motion_type": "wellness",
            "quality_score": 0.85
        },
        
        # Complex combination motions
        {
            "id": "combo_001",
            "name": "Parkour Obstacle Course",
            "description": "Complex parkour sequence with vaulting, climbing, jumping, and fluid transitions between obstacles",
            "motion_vector": [0.85, 0.9, 0.8, 0.87, 0.82, 0.9, 0.88, 0.83],
            "metadata": {
                "category": "athletic",
                "tags": ["parkour", "obstacles", "vaulting", "climbing", "fluid", "complex"],
                "duration": 25.3,
                "frames": 759,
                "fps": 30,
                "complexity": 0.95
            },
            "motion_type": "athletic",
            "quality_score": 0.93
        }
    ]

def test_semantic_search_scenarios():
    """Test various semantic search scenarios for motion capture data."""
    print("🎭 Testing Motion Capture Semantic Search Scenarios")
    print("=" * 65)
    
    try:
        from app import app
        print("✅ Flask app imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        return
    
    # Test scenarios with expected results
    test_scenarios = [
        {
            "name": "Athletic Performance Search",
            "query": "high-energy athletic performance with explosive movements",
            "expected_categories": ["athletic"],
            "expected_keywords": ["explosive", "athletic", "sports"]
        },
        {
            "name": "Graceful Movement Search", 
            "query": "graceful flowing movements with artistic expression",
            "expected_categories": ["dance", "wellness"],
            "expected_keywords": ["graceful", "flowing", "artistic"]
        },
        {
            "name": "Professional/Business Context",
            "query": "professional business interaction with confident posture",
            "expected_categories": ["gesture"],
            "expected_keywords": ["professional", "business", "confident"]
        },
        {
            "name": "Combat/Martial Arts",
            "query": "martial arts combat with precise strikes and discipline", 
            "expected_categories": ["martial-arts"],
            "expected_keywords": ["martial", "combat", "precise", "discipline"]
        },
        {
            "name": "Everyday Natural Movement",
            "query": "natural everyday movement with casual relaxed style",
            "expected_categories": ["locomotion", "gesture"],
            "expected_keywords": ["natural", "everyday", "casual", "relaxed"]
        },
        {
            "name": "Performance and Theater",
            "query": "theatrical performance with dramatic expression",
            "expected_categories": ["performance"],
            "expected_keywords": ["theater", "dramatic", "performance"]
        },
        {
            "name": "Wellness and Mindfulness",
            "query": "peaceful mindful movement for wellness and meditation",
            "expected_categories": ["wellness"],
            "expected_keywords": ["peaceful", "mindful", "wellness", "meditation"]
        },
        {
            "name": "Complex Multi-skill Movement",
            "query": "complex athletic sequence with multiple skills and transitions",
            "expected_categories": ["athletic"],
            "expected_keywords": ["complex", "sequence", "transitions", "obstacles"]
        }
    ]
    
    # Create test data
    test_motions = create_motion_test_data()
    print(f"📊 Created {len(test_motions)} test motion samples")
    
    with app.test_client() as client:
        # First, bulk index the test data
        print(f"\n📥 Indexing test motion data...")
        response = client.post('/index/bulk',
                             data=json.dumps(test_motions),
                             content_type='application/json')
        
        if response.status_code == 200:
            result = response.get_json()
            print(f"✅ Indexed {result.get('indexed', 0)} motions")
        else:
            print(f"⚠️ Indexing in mock mode: {response.get_json()}")
        
        # Wait a moment for indexing (in real scenarios)
        time.sleep(1)
        
        print(f"\n🔍 Running Semantic Search Scenarios")
        print("-" * 45)
        
        all_passed = True
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   Query: \"{scenario['query']}\"")
            
            # Test semantic search
            response = client.post('/search/semantic',
                                 data=json.dumps({
                                     "query": scenario['query'],
                                     "k": 5
                                 }),
                                 content_type='application/json')
            
            if response.status_code == 200:
                results = response.get_json()
                search_results = results.get('results', [])
                
                print(f"   Results: {len(search_results)} motions found")
                
                # Analyze results
                found_categories = set()
                found_keywords = set()
                
                for result in search_results[:3]:  # Check top 3 results
                    category = result.get('metadata', {}).get('category', '')
                    tags = result.get('metadata', {}).get('tags', [])
                    name = result.get('name', '').lower()
                    description = result.get('description', '').lower()
                    
                    found_categories.add(category)
                    found_keywords.update(tags)
                    
                    # Add keywords from name and description
                    for keyword in scenario['expected_keywords']:
                        if keyword.lower() in name or keyword.lower() in description:
                            found_keywords.add(keyword)
                    
                    score = result.get('semantic_score', 0)
                    print(f"     - {result.get('name', 'Unknown')} (category: {category}, score: {score:.3f})")
                
                # Evaluate scenario success
                category_match = any(cat in found_categories for cat in scenario['expected_categories'])
                keyword_match = any(kw.lower() in [fk.lower() for fk in found_keywords] for kw in scenario['expected_keywords'])
                
                if category_match or keyword_match:
                    print(f"   ✅ Scenario PASSED")
                    if category_match:
                        print(f"      Found expected categories: {list(found_categories & set(scenario['expected_categories']))}")
                    if keyword_match:
                        matching_keywords = [kw for kw in scenario['expected_keywords'] 
                                           if kw.lower() in [fk.lower() for fk in found_keywords]]
                        print(f"      Found expected keywords: {matching_keywords}")
                else:
                    print(f"   ❌ Scenario FAILED")
                    print(f"      Expected categories: {scenario['expected_categories']}")
                    print(f"      Found categories: {list(found_categories)}")
                    print(f"      Expected keywords: {scenario['expected_keywords']}")
                    print(f"      Found keywords: {list(found_keywords)}")
                    all_passed = False
            else:
                print(f"   ❌ Search failed: {response.get_json()}")
                all_passed = False
        
        # Test hybrid search scenarios
        print(f"\n🔀 Testing Hybrid Search Integration")
        print("-" * 35)
        
        hybrid_test = {
            "vector": [0.8, 0.9, 0.7, 0.8, 0.9, 0.6, 0.8, 0.7],  # High-energy vector
            "query": "explosive athletic movement",
            "k": 3,
            "vector_weight": 0.6
        }
        
        response = client.post('/search/hybrid',
                             data=json.dumps(hybrid_test),
                             content_type='application/json')
        
        if response.status_code == 200:
            results = response.get_json()
            print(f"✅ Hybrid search successful")
            print(f"   Vector weight: {results.get('weights', {}).get('vector', 0)}")
            print(f"   Text weight: {results.get('weights', {}).get('text', 0)}")
            print(f"   Results: {results.get('total', 0)}")
            
            for i, result in enumerate(results.get('results', [])[:3], 1):
                print(f"   {i}. {result.get('name', 'Unknown')} (score: {result.get('hybrid_score', 0):.3f})")
        else:
            print(f"❌ Hybrid search failed: {response.get_json()}")
            all_passed = False
        
        # Performance and edge case tests
        print(f"\n⚡ Testing Performance and Edge Cases")
        print("-" * 35)
        
        edge_cases = [
            {"query": "", "expected_error": True, "name": "Empty query"},
            {"query": "nonexistent超级稀有motion", "expected_results": 0, "name": "Non-existent terms"},
            {"query": "a " * 100, "expected_results": 0, "name": "Very long query"},
            {"query": "dance athletic martial wellness", "expected_results": 0, "name": "Multi-category query"}
        ]
        
        for case in edge_cases:
            response = client.post('/search/semantic',
                                 data=json.dumps({
                                     "query": case['query'],
                                     "k": 5
                                 }),
                                 content_type='application/json')
            
            if case.get('expected_error'):
                success = response.status_code != 200
                print(f"   {case['name']}: {'✅' if success else '❌'} Expected error")
            else:
                if response.status_code == 200:
                    results = response.get_json()
                    result_count = results.get('total', 0)
                    
                    if case['expected_results'] == 0:
                        success = result_count == 0
                    elif case['expected_results'] == ">0":
                        success = result_count > 0
                    else:
                        success = True  # Just check it doesn't error
                    
                    print(f"   {case['name']}: {'✅' if success else '❌'} Results: {result_count}")
                else:
                    print(f"   {case['name']}: ❌ Unexpected error")
                    all_passed = False
        
        print(f"\n🎉 Motion Capture Semantic Search Testing Complete!")
        print("=" * 65)
        print(f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return all_passed

if __name__ == '__main__':
    test_semantic_search_scenarios()