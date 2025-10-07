import React, {useState} from 'react'
import {startBlend} from '../client'

export default function BlendControls({selected}: any){
  const [weight, setWeight] = useState(0.5)
  const [status, setStatus] = useState<string | null>(null)

  async function onStart(){
    if(selected.length === 0){
      setStatus('Select at least one motion')
      return
    }
    setStatus('Submitting...')
    try{
      const res = await startBlend({input_motions: selected, params: {blend_weight: weight}})
      setStatus(`Job submitted: ${res.job_id}`)
    }catch(e:any){
      setStatus('Failed to submit: '+(e?.message||String(e)))
    }
  }

  return (
    <div className="blend-controls">
      <h3>Blend Controls</h3>
      <div>
        <label>Blend weight: {weight}</label>
        <input type="range" min={0} max={1} step={0.01} value={weight} onChange={e=>setWeight(Number(e.target.value))} />
      </div>
      <div className="actions">
        <button onClick={onStart}>Start Blend</button>
      </div>
      {status && <div className="status">{status}</div>}
    </div>
  )
}
