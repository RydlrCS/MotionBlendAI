import React, {useState} from 'react'
import {describeArtifact} from '../client'

export default function ArtifactsList({artifacts}: any){
  const [desc, setDesc] = useState<Record<string, any> | null>(null)
  const [loading, setLoading] = useState(false)

  async function onClick(name:string){
    setLoading(true)
    try{
      const d = await describeArtifact(name)
      setDesc({name, ...d})
    }catch(e:any){
      setDesc({name, error: e?.message || String(e)})
    }finally{ setLoading(false) }
  }

  return (
    <div className="artifacts">
      <h3>Artifacts</h3>
      {artifacts.length === 0 ? <div>No artifacts</div> : (
        <div style={{display:'flex', gap:12}}>
          <ul>
            {artifacts.map((a: string)=> (
              <li key={a}><button onClick={()=>onClick(a)}>{a}</button></li>
            ))}
          </ul>
          <div className="artifact-desc">
            {loading && <div>Loading...</div>}
            {desc && (
              <div>
                <h4>{desc.name}</h4>
                {desc.error ? <pre>{String(desc.error)}</pre> : (
                  <div>
                    <div>shape: {JSON.stringify(desc.shape)}</div>
                    <div>dtype: {desc.dtype}</div>
                    <div>sample: {JSON.stringify(desc.sample)}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
