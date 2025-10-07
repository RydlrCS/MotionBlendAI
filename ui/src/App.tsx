import React, {useState, useEffect} from 'react'
import MotionList from './components/MotionList'
import BlendControls from './components/BlendControls'
import {getMotions, getArtifacts} from './client'

const ArtifactsList: React.FC<{artifacts: string[]}> = ({artifacts}) => {
  return (
    <div className="artifacts-list">
      {artifacts.length === 0 ? (
        <div className="empty">No artifacts</div>
      ) : (
        <ul>
          {artifacts.map((a, idx) => (
            <li key={idx}>{a}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function App(){
  const [motions, setMotions] = useState([])
  const [selected, setSelected] = useState<string[]>([])

  useEffect(()=>{
    let mounted = true
    getMotions().then(data=>{
      if(mounted) setMotions(data.motions || [])
    }).catch(()=>setMotions([]))
    return ()=>{ mounted=false }
  },[])

  const [artifacts, setArtifacts] = useState<string[]>([])
  useEffect(()=>{
    let mounted = true
    getArtifacts().then(d=>{ if(mounted) setArtifacts(d.artifacts || []) }).catch(()=>{})
    return ()=>{ mounted=false }
  },[])

  return (
    <div className="app">
      <header className="header">MotionBlend AI - Demo</header>
      <div className="content">
        <aside className="left">
          <MotionList motions={motions} selected={selected} onToggle={(id: string)=>{
            setSelected(prev => prev.includes(id)? prev.filter(x=>x!==id) : [...prev, id])
          }}/>
        </aside>
        <main className="main">
          <BlendControls selected={selected} />
          <ArtifactsList artifacts={artifacts} />
        </main>
      </div>
    </div>
  )
}
