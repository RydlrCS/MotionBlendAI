import React, {useState, memo} from 'react'
import {describeArtifact, getArtifactAnalysis} from '../client'
import Sparkline from './Sparkline'
import MotionBlendAnalysis from './MotionBlendAnalysis'
import MotionStrip from './MotionStrip'
import type { Artifact, Segment } from '../types/artifacts'

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
  const [analysisData, setAnalysisData] = useState<any>(null)

  async function onClick(item: any){
    setLoading(true)
    const name = item.name || item.id
    try{
      // If item already has full artifact data (blend artifact), use it directly
      if (item.type === 'motion_blend') {
        setDesc({name, artifact: item})
        
        // Try to fetch detailed analysis if not already in artifact
        if (!item.analysis) {
          const analysis = await getArtifactAnalysis(item.id || name)
          if (analysis) {
            setAnalysisData(analysis)
            // Merge analysis into the artifact
            item.analysis = analysis.analysis
            item.metadata.quality_score = analysis.analysis?.metrics?.quality_score
            item.metadata.quality_category = analysis.analysis?.metrics?.quality_category
          }
        } else {
          setAnalysisData({ analysis: item.analysis })
        }
      } else {
        const d = await describeArtifact(name)
        setDesc({name, ...d})
        setAnalysisData(null)
      }
    }catch(e:any){
      setDesc({name, error: e?.message || String(e)})
      setAnalysisData(null)
    }finally{ setLoading(false) }
  }

  function humanSize(n?: number){
    if(n === undefined || n === null) return ''
    if(n < 1024) return n + ' B'
    if(n < 1024*1024) return (n/1024).toFixed(1) + ' KB'
    return (n/(1024*1024)).toFixed(2) + ' MB'
  }

  // Extract artifacts from manifest or use direct artifacts array
  const items = manifest?.artifacts || artifacts || []

  return (
    <div className="artifacts">
      <h3>Artifacts</h3>
      {(!items || items.length === 0) ? <div>No artifacts</div> : (
        <div style={{display:'flex', gap:12}}>
          <ul>
            {items.map((a: any)=> (
              <li key={a.id || a.name} style={{marginBottom:8}}>
                <div style={{display:'flex', gap:8, alignItems:'center'}}>
                  <button onClick={()=>onClick(a)}>{a.name}</button>
                  <small style={{color:'#666'}}>
                    {a.type ? a.type.toUpperCase() : (a.kind ? a.kind.toUpperCase() : '')} 
                    {a.metadata?.size ? `· ${a.metadata.size}` : (a.size ? `· ${humanSize(a.size)}` : '')}
                  </small>
                  {a.file_path && (
                    <a style={{marginLeft:8}} href={`/artifacts/${encodeURIComponent(a.name)}`} target="_blank" rel="noopener noreferrer">Download</a>
                  )}
                  {/* Quality badge for motion blends */}
                  {a.type === 'motion_blend' && a.metadata?.quality_category && (
                    <span 
                      className={`quality-badge ${a.metadata.quality_category}`}
                      title={`Quality Score: ${(a.metadata.quality_score * 100).toFixed(0)}%`}
                    >
                      {a.metadata.quality_category}
                    </span>
                  )}
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
                    {/* Motion Blend Analysis - show for blend artifacts */}
                    {desc.artifact?.type === 'motion_blend' && (
                      <div style={{marginBottom: 24}}>
                        {/* Motion Strips Visualization */}
                        {desc.artifact.sources && desc.artifact.blend && (
                          <div className="motion-strips-section" style={{marginBottom: 32}}>
                              <h5 style={{marginBottom: 16, fontSize: '16px', fontWeight: 600}}>🎬 Motion Strips</h5>
                              
                              {/* Source Motion Strips */}
                              {desc.artifact.sources.map((source: any, idx: number) => (
                                <MotionStrip
                                  key={source.id}
                                  label={`Source ${idx + 1}: ${source.label}`}
                                  thumbnails={source.thumbnails || []}
                                  band={[{
                                    fromFrame: 0,
                                    toFrame: source.frames - 1,
                                    label: source.label,
                                    color: source.color,
                                    alpha: 1
                                  }] as Segment[]}
                                  fps={desc.artifact.fps || 30}
                                  frameStep={source.sampleEvery || 10}
                                  highlightWindows={[]}
                                  emphasis={false}
                                />
                              ))}
                              
                              {/* Blend Motion Strip */}
                              {desc.artifact.blend && (
                                <MotionStrip
                                  label={`Blend Result: ${desc.artifact.blend.label || desc.artifact.name}`}
                                  thumbnails={desc.artifact.blend.thumbnails || []}
                                  band={desc.artifact.blend.segments || []}
                                  fps={desc.artifact.fps || 30}
                                  frameStep={desc.artifact.blend.sampleEvery || 10}
                                  highlightWindows={desc.artifact.metrics?.transitionWindows || []}
                                  emphasis={true}
                                  metrics={desc.artifact.metrics}
                                />
                            )}
                          </div>
                        )}                        {/* Metrics Summary Panel */}
                        {(desc.artifact.analysis || analysisData) && (
                          <div className="metrics-summary">
                            <h5>📊 Analysis Metrics</h5>
                            <div className="metrics-grid">
                              {(() => {
                                const metrics = desc.artifact.analysis?.metrics || analysisData?.analysis?.metrics
                                if (!metrics) return null
                                
                                return (
                                  <>
                                    <div className="metric-card">
                                      <span className="metric-label">Mean Velocity</span>
                                      <span className="metric-value">{metrics.mean_velocity?.toFixed(3)}</span>
                                    </div>
                                    <div className="metric-card">
                                      <span className="metric-label">Mean Acceleration</span>
                                      <span className="metric-value">{metrics.mean_acceleration?.toFixed(3)}</span>
                                    </div>
                                    <div className="metric-card">
                                      <span className="metric-label">Transition Smoothness</span>
                                      <span className="metric-value">{metrics.transition_smoothness?.toFixed(3)}</span>
                                    </div>
                                    <div className="metric-card">
                                      <span className="metric-label">Global Diversity</span>
                                      <span className="metric-value">{metrics.global_diversity?.toFixed(3)}</span>
                                    </div>
                                    <div className="metric-card">
                                      <span className="metric-label">Quality Score</span>
                                      <span className={`metric-value quality-${metrics.quality_category}`}>
                                        {(metrics.quality_score * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                  </>
                                )
                              })()}
                            </div>
                          </div>
                        )}
                        
                        {/* Velocity/Acceleration Graphs */}
                        <MotionBlendAnalysis artifact={desc.artifact} width={600} height={300} />
                      </div>
                    )}
                    
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
