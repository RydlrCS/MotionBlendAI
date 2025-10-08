import axios from 'axios'

const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:3000'

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
