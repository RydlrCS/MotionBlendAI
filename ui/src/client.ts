import axios from 'axios'

// Use environment variables with fallback
const API_BASE = import.meta.env?.VITE_API_URL || 'http://localhost:5000'
const ELASTIC_API_BASE = import.meta.env?.VITE_API_URL || 'http://localhost:5000'

// Search interface
interface SearchQuery {
  vector: number[]
  k?: number
}

export async function getMotions(){
  try {
    const res = await axios.get(`${API_BASE}/motions`)
    return { motions: res.data }
  } catch (error) {
    console.error('Failed to fetch motions:', error)
    // Return mock data for development
    return {
      motions: [
        {
          id: 'mock_001',
          name: 'Walking Forward',
          metadata: { category: 'locomotion', duration: 2.5, frames: 75 }
        },
        {
          id: 'mock_002', 
          name: 'Jump Landing',
          metadata: { category: 'athletic', duration: 1.8, frames: 54 }
        }
      ]
    }
  }
}

export async function startBlend(payload:any){
  try {
    const res = await axios.post(`${API_BASE}/api/blend`, payload)
    return res.data
  } catch (error) {
    console.error('Blend API error:', error)
    // Return mock blend result for development
    const mockResult = {
      id: `blend_${Date.now()}`,
      name: `${payload.motion1}_${payload.motion2}_blend`,
      weight: payload.weight,
      status: 'completed',
      created_at: new Date().toISOString(),
      metadata: {
        source_motions: [payload.motion1, payload.motion2],
        blend_weight: payload.weight,
        frames: 120,
        duration: 4.0
      }
    }
    return mockResult
  }
}

export async function getArtifacts(){
  const res = await axios.get(`${API_BASE}/api/artifacts`)
  return res.data
}

export async function getArtifactsManifest(){
  try {
    const res = await axios.get(`${API_BASE}/api/artifacts/manifest`)
    return res.data
  } catch (error) {
    console.error('Artifacts manifest error:', error)
    // Return mock manifest for development
    return {
      artifacts: [],
      total: 0,
      last_updated: new Date().toISOString()
    }
  }
}

export async function describeArtifact(name:string){
  const res = await axios.get(`${API_BASE}/api/artifact/${encodeURIComponent(name)}/describe`)
  return res.data
}

// Search motions using Elasticsearch vector search
export async function searchMotions(query: SearchQuery): Promise<any[]> {
  try {
    const response = await axios.post(`${ELASTIC_API_BASE}/search/vector`, query)
    return response.data.results || response.data
  } catch (error) {
    console.error('Elasticsearch search failed:', error)
    // Return mock results for development
    return [
      {
        id: 'search_001',
        name: 'Similar Motion 1',
        vector: query.vector,
        metadata: {
          frames: 150,
          joints: 25,
          duration: 5.0,
          format: 'FBX'
        },
        similarity_score: 0.92
      }
    ]
  }
}
