import axios from 'axios'

const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:3000'
const ELASTIC_API_BASE = 'http://localhost:5002';

// Search interface
interface SearchQuery {
  vector: number[]
  k?: number
}

export async function getMotions(){
  const res = await axios.get(`${API_BASE}/api/motions`)
  return res.data
}

export async function startBlend(payload:any){
  const res = await axios.post(`${API_BASE}/api/blend`, payload)
  return res.data
}

export async function getArtifacts(){
  const res = await axios.get(`${API_BASE}/api/artifacts`)
  return res.data
}

export async function getArtifactsManifest(){
  const res = await axios.get(`${API_BASE}/api/artifacts/manifest`)
  return res.data
}

export async function describeArtifact(name:string){
  const res = await axios.get(`${API_BASE}/api/artifact/${encodeURIComponent(name)}/describe`)
  return res.data
}

// Search motions using Elasticsearch vector search
export async function searchMotions(query: SearchQuery): Promise<any[]> {
  try {
    const response = await axios.post(`${ELASTIC_API_BASE}/search`, query)
    return response.data
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
