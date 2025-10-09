/**
 * Main Application Component - OBS-Style Motion Blend Interface
 * 
 * Provides a professional video editor-style interface for motion capture
 * blending with tabs for Motion Mixer and Artifact Inspector.
 */
import React, {useState, useEffect} from 'react'
import MotionList from './components/MotionList'
import BlendControls from './components/BlendControls'
import ArtifactsList from './components/ArtifactsList'
import BlendMixer from './components/BlendMixer'
import {getMotions, getArtifactsManifest, startBlend} from './client'

/**
 * Main App component with OBS-style interface
 * 
 * Features:
 * - Professional dark theme
 * - Tab-based navigation (Motion Mixer / Artifacts)
 * - Integrated blend request handling
 * - Real-time artifact manifest updates
 */
export default function App(){
  // Motion data state
  const [motions, setMotions] = useState([])
  const [selected, setSelected] = useState<string[]>([])
  
  // Navigation state
  const [view, setView] = useState<'mixer'|'artifacts'>('mixer')

  // Load available motions from API
  useEffect(()=>{
    let mounted = true
    getMotions().then(data=>{
      if(mounted) setMotions(data.motions || [])
    }).catch(()=>setMotions([]))
    return ()=>{ mounted=false }
  },[])

  // Artifact manifest state
  const [manifest, setManifest] = useState<any>(null)
  useEffect(()=>{
    let mounted = true
    getArtifactsManifest().then(d=>{
      if(!mounted) return
      setManifest(d)
    }).catch(()=>{})
    return ()=>{ mounted=false }
  },[])

  /**
   * Handle blend operation requests from BlendMixer
   * 
   * Initiates motion blending via API and refreshes artifacts
   * 
   * @param seq1 First sequence data
   * @param seq2 Second sequence data 
   * @param weight Blend weight (0.0-1.0)
   */
  const handleBlendRequest = async (seq1: any, seq2: any, weight: number) => {
    try {
      const result = await startBlend({
        motion1: seq1.name,
        motion2: seq2.name, 
        weight: weight
      })
      console.log('Blend request submitted:', result)
      
      // Refresh manifest after blend operation
      setTimeout(() => {
        getArtifactsManifest().then(setManifest).catch(() => {})
      }, 1000)
    } catch (e) {
      console.error('Blend failed:', e)
    }
  }

  return (
    <div className="app">
      {/* Professional header with app title */}
      <header className="header">
        MotionBlend AI - OBS Style Mixer
      </header>
      
      {/* Tab navigation for different views */}
      <div className="nav-tabs">
        <button 
          className={`nav-tab ${view === 'mixer' ? 'active' : ''}`}
          onClick={() => setView('mixer')}
          title="Motion Capture Blending Interface"
        >
          Motion Mixer
        </button>
        <button 
          className={`nav-tab ${view === 'artifacts' ? 'active' : ''}`}
          onClick={() => setView('artifacts')}
          title="View Generated Artifacts"
        >
          Artifacts
        </button>
      </div>
      
      {/* Main content area - switches based on active tab */}
      <div className="content">
        {view === 'mixer' ? (
          /* OBS-style motion blending interface */
          <BlendMixer onBlendRequest={handleBlendRequest} />
        ) : (
          /* Artifact inspector for viewing generated content */
          <main className="main">
            <ArtifactsList manifest={manifest} />
          </main>
        )}
      </div>
    </div>
  )
}
