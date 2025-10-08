import React, {useState, useEffect} from 'react'
import MotionList from './components/MotionList'
import BlendControls from './components/BlendControls'
import ArtifactsList from './components/ArtifactsList'
import {getMotions, getArtifactsManifest} from './client'

export default function App(){
  const [motions, setMotions] = useState([])
  const [selected, setSelected] = useState<string[]>([])
  const [view, setView] = useState<'motions'|'artifacts'>('motions')

  useEffect(()=>{
    let mounted = true
    getMotions().then(data=>{
      if(mounted) setMotions(data.motions || [])
    }).catch(()=>setMotions([]))
    return ()=>{ mounted=false }
  },[])

  const [manifest, setManifest] = useState<any>(null)
  useEffect(()=>{
    let mounted = true
    getArtifactsManifest().then(d=>{
      if(!mounted) return
      setManifest(d)
    }).catch(()=>{})
    return ()=>{ mounted=false }
  },[])

  return (
    <div className="app">
      <header className="header">MotionBlend AI - Demo</header>
      <div style={{padding:8, display:'flex', gap:8}}>
        <button onClick={()=>setView('motions')} style={{fontWeight: view==='motions'? '600':'400'}}>Motions</button>
        <button onClick={()=>setView('artifacts')} style={{fontWeight: view==='artifacts'? '600':'400'}}>Artifacts</button>
      </div>
      <div className="content">
        <aside className="left">
          <MotionList motions={motions} selected={selected} onToggle={(id: string)=>{
            setSelected(prev => prev.includes(id)? prev.filter(x=>x!==id) : [...prev, id])
          }}/>
        </aside>
        <main className="main">
          {view === 'motions' ? (
            <>
              <BlendControls selected={selected} />
              <ArtifactsList manifest={manifest} />
            </>
          ) : (
            <ArtifactsList manifest={manifest} />
          )}
        </main>
      </div>
    </div>
  )
}
