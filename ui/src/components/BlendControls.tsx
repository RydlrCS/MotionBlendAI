import React, {useState} from 'react'
import {startBlend} from '../client'

interface BlendControlsProps {
  selected: string[]
  motions: any[]
}

export default function BlendControls({selected, motions}: BlendControlsProps){
  const [weight, setWeight] = useState(0.5)
  const [status, setStatus] = useState<string | null>(null)

  // Get selected motion objects
  const selectedMotions = motions.filter(motion => selected.includes(motion.id))

  async function onStart(){
    if(selected.length === 0){
      setStatus('Select at least two motions to blend')
      return
    }
    if(selected.length < 2){
      setStatus('Select at least two motions to blend')
      return
    }
    setStatus('Submitting blend request...')
    try{
      const res = await startBlend({
        input_motions: selected, 
        params: {blend_weight: weight}
      })
      setStatus(`✅ Blend job submitted: ${res.job_id || 'Processing...'}`)
    }catch(e:any){
      setStatus(`❌ Failed to submit: ${e?.message || String(e)}`)
    }
  }

  return (
    <div className="blend-controls">
      <h3>Motion Blend Mixer</h3>
      
      {/* Selected Motions Display */}
      <div className="selected-motions">
        <h4>Selected Motions ({selected.length})</h4>
        {selectedMotions.length === 0 ? (
          <div className="motion-slot empty">
            <p>Select motions from the list to begin blending</p>
          </div>
        ) : (
          <div className="motion-slots">
            {selectedMotions.map((motion, index) => (
              <div key={motion.id} className="motion-slot filled">
                <div className="motion-info">
                  <span className="motion-name">{motion.name}</span>
                  <span className="motion-meta">
                    {motion.metadata?.category || 'Unknown'} • 
                    {motion.metadata?.duration?.toFixed(1) || '0.0'}s
                  </span>
                </div>
                <span className="slot-label">Motion {index + 1}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Blend Weight Control */}
      <div className="weight-control">
        <label>Blend Weight: <strong>{weight.toFixed(2)}</strong></label>
        <input 
          type="range" 
          min={0} 
          max={1} 
          step={0.01} 
          value={weight} 
          onChange={e=>setWeight(Number(e.target.value))}
          className="weight-slider"
        />
        <div className="weight-labels">
          <span>Motion A (100%)</span>
          <span>Motion B (100%)</span>
        </div>
      </div>

      {/* Action Button */}
      <div className="actions">
        <button 
          onClick={onStart}
          className="blend-button"
          disabled={selected.length < 2}
        >
          {selected.length < 2 ? 'Select 2+ Motions' : 'Start Blend Process'}
        </button>
      </div>
      
      {/* Status Display */}
      {status && (
        <div className={`status ${status.includes('✅') ? 'success' : status.includes('❌') ? 'error' : 'info'}`}>
          {status}
        </div>
      )}
    </div>
  )
}
