import React, {useState} from 'react'
import {describeArtifact} from '../client'
import Sparkline from './Sparkline'

// Allow the custom <model-viewer> element in JSX/TSX
declare global {
  namespace JSX {
    interface IntrinsicElements {
      'model-viewer': any
    }
  }
}

export default function ArtifactsList({artifacts, manifest}: any){
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

  function humanSize(n?: number){
    if(n === undefined || n === null) return ''
    if(n < 1024) return n + ' B'
    if(n < 1024*1024) return (n/1024).toFixed(1) + ' KB'
    return (n/(1024*1024)).toFixed(2) + ' MB'
  }

  const items = manifest ? (
    [
      ...Object.entries(manifest.inputs || {}).map(([k,v])=>({name:k,size:v, kind:'input'})),
      ...(manifest.outputs || []).map((o:any)=>({name:o.name,size:o.size, kind:'output'}))
    ]
  ) : (artifacts || [])

  return (
    <div className="artifacts">
      <h3>Artifacts</h3>
      {(!items || items.length === 0) ? <div>No artifacts</div> : (
        <div style={{display:'flex', gap:12}}>
          <ul>
            {items.map((a: any)=> (
              <li key={a.name} style={{marginBottom:8}}>
                <div style={{display:'flex', gap:8, alignItems:'center'}}>
                  <button onClick={()=>onClick(a.name)}>{a.name}</button>
                  <small style={{color:'#666'}}>{a.kind ? a.kind.toUpperCase() : ''} {a.size ? `· ${humanSize(a.size)}` : ''}</small>
                  <a style={{marginLeft:8}} href={`/artifacts/${encodeURIComponent(a.name)}`} target="_blank" rel="noopener noreferrer">Download</a>
                </div>
              </li>
            ))}
          </ul>
          <div className="artifact-desc">
            {loading && <div>Loading...</div>}
            {desc && (
              <div>
                <h4>{desc.name} {desc.size ? <small>({desc.size} bytes)</small> : null}</h4>
                <div style={{marginBottom:8}}>
                  <a href={`/artifacts/${encodeURIComponent(desc.name)}`} target="_blank" rel="noopener noreferrer"><button>Download</button></a>
                </div>
                {desc.error ? <pre>{String(desc.error)}</pre> : (
                  <div>
                    {/* If GLB, show 3D preview */}
                    {desc.name && desc.name.toLowerCase().endsWith('.glb') ? (
                      <div>
                        <model-viewer src={`/artifacts/${encodeURIComponent(desc.name)}`} alt={desc.name} camera-controls auto-rotate style={{width:300, height:200}}></model-viewer>
                      </div>
                    ) : (
                      <div>
                        <div>shape: {JSON.stringify(desc.shape)}</div>
                        <div>dtype: {desc.dtype}</div>
                        <div>sample: {JSON.stringify(desc.sample)}</div>
                        {Array.isArray(desc.sample) && desc.sample.length > 0 && (
                          <div style={{marginTop:8}}>
                            <strong>Preview</strong>
                            <div><Sparkline values={desc.sample} width={300} height={50} /></div>
                          </div>
                        )}
                      </div>
                    )}
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
