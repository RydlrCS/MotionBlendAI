/**
 * BlendMixer - OBS Studio-style Motion Blending Interface
 * 
 * This component provides a professional video mixer-like interface for blending
 * two motion capture sequences. Features include:
 * - Dual sequence viewers with synchronized timeline scrubbing
 * - Real-time blend weight adjustment
 * - Overlay visualization of blend preview
 * - Motion sequence selection and metadata display
 * 
 * Key Features:
 * - Side-by-side sequence comparison
 * - Frame-accurate timeline synchronization
 * - Visual blend weight slider (0.0 = 100% Sequence A, 1.0 = 100% Sequence B)
 * - Optional blend overlay preview
 * - Professional OBS-style layout and controls
 */
import React, {useState, useEffect} from 'react'
import SequenceViewer from './SequenceViewer'
import {getMotions} from '../client'

/**
 * Motion sequence data structure
 * Contains all metadata needed for visualization and blending
 */
interface SequenceData {
  name: string          // Display name of the motion
  shape: number[]       // [frames, joints, dimensions] array shape
  frames: number        // Total number of animation frames
  joints: number        // Number of skeleton joints
  duration: number      // Sequence duration in seconds
  data: any            // Raw motion data from API
}

/**
 * BlendMixer component props
 * Callback-based interface for triggering blend operations
 */
interface BlendMixerProps {
  /** Callback fired when user requests a blend operation */
  onBlendRequest?: (seq1: SequenceData, seq2: SequenceData, weight: number) => void
}

/**
 * Main BlendMixer component implementation
 * 
 * State Management:
 * - motions: Available motion sequences from API
 * - sequence1/sequence2: Currently selected sequences for blending
 * - currentFrame: Synchronized timeline position (0-based index)
 * - blendWeight: Blend ratio between sequences (0.0-1.0)
 * - showOverlay: Toggle for blend preview visualization
 */
export default function BlendMixer({onBlendRequest}: BlendMixerProps) {
  // Motion sequence management
  const [motions, setMotions] = useState<any[]>([])                    // Available motions from API
  const [sequence1, setSequence1] = useState<SequenceData | null>(null) // Selected sequence A
  const [sequence2, setSequence2] = useState<SequenceData | null>(null) // Selected sequence B
  
  // Timeline and playback state
  const [currentFrame, setCurrentFrame] = useState(0)                  // Current timeline position (0-based)
  
  // Blend control state
  const [blendWeight, setBlendWeight] = useState(0.5)                  // Blend weight: 0.0=100% seq1, 1.0=100% seq2
  const [showOverlay, setShowOverlay] = useState(false)                // Show/hide blend preview overlay

  useEffect(() => {
    getMotions().then(data => {
      setMotions(data.motions || [])
    }).catch(() => setMotions([]))
  }, [])

  /**
   * Convert raw motion API data to normalized sequence data structure
   * 
   * This function transforms various motion data formats into a consistent
   * SequenceData interface that the UI components can work with.
   * 
   * @param motion Raw motion data from API
   * @returns Normalized SequenceData object
   */
  const createSequenceData = (motion: any): SequenceData => {
    return {
      name: motion.name || motion.id,                           // Display name
      shape: motion.shape || [100, 3],                         // [frames, joints, dims] or fallback
      frames: motion.shape?.[0] || 100,                        // Total animation frames
      joints: motion.shape?.[1] || 3,                          // Number of skeleton joints
      duration: (motion.shape?.[0] || 100) / 30,               // Duration at 30fps
      data: motion                                              // Keep original data for blending
    }
  }

  /**
   * Calculate the maximum frame count between both sequences
   * Used for timeline synchronization and boundary checking
   */
  const maxFrames = Math.max(
    sequence1?.frames || 0,
    sequence2?.frames || 0
  )

  /**
   * Handle blend operation request
   * 
   * Validates that both sequences are selected and triggers the
   * onBlendRequest callback with current blend parameters.
   * This initiates the actual motion blending computation.
   */
  const handleBlend = () => {
    if (sequence1 && sequence2 && onBlendRequest) {
      onBlendRequest(sequence1, sequence2, blendWeight)
    }
  }

  return (
    <div className="blend-mixer">
      <div className="mixer-header">
        <h2>Motion Blend Mixer</h2>
        <div className="mixer-controls">
          <label>
            <input 
              type="checkbox" 
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
            />
            Show Blend Overlay
          </label>
        </div>
      </div>

      <div className="sequence-selectors">
        <div className="selector">
          <label>Sequence A:</label>
          <select 
            value={sequence1?.name || ''}
            onChange={(e) => {
              const motion = motions.find(m => (m.name || m.id) === e.target.value)
              setSequence1(motion ? createSequenceData(motion) : null)
              setCurrentFrame(0)
            }}
          >
            <option value="">Select motion...</option>
            {motions.map(motion => (
              <option key={motion.id || motion.name} value={motion.name || motion.id}>
                {motion.name || motion.id}
              </option>
            ))}
          </select>
        </div>
        
        <div className="selector">
          <label>Sequence B:</label>
          <select 
            value={sequence2?.name || ''}
            onChange={(e) => {
              const motion = motions.find(m => (m.name || m.id) === e.target.value)
              setSequence2(motion ? createSequenceData(motion) : null)
              setCurrentFrame(0)
            }}
          >
            <option value="">Select motion...</option>
            {motions.map(motion => (
              <option key={motion.id || motion.name} value={motion.name || motion.id}>
                {motion.name || motion.id}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="dual-viewer">
        <SequenceViewer 
          sequence={sequence1}
          currentFrame={currentFrame}
          onFrameChange={setCurrentFrame}
          title="Sequence A"
          isActive={true}
        />
        
        <SequenceViewer 
          sequence={sequence2}
          currentFrame={currentFrame}
          onFrameChange={setCurrentFrame}
          title="Sequence B"
          isActive={true}
        />
      </div>

      {showOverlay && sequence1 && sequence2 && (
        <div className="blend-overlay">
          <h4>Blend Preview (Frame {currentFrame + 1})</h4>
          <div className="blend-preview">
            Weight: A({(1-blendWeight).toFixed(2)}) + B({blendWeight.toFixed(2)})
          </div>
        </div>
      )}

      <div className="blend-controls">
        <div className="weight-control">
          <label>Blend Weight:</label>
          <input 
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={blendWeight}
            onChange={(e) => setBlendWeight(parseFloat(e.target.value))}
          />
          <span>{blendWeight.toFixed(2)}</span>
        </div>
        
        <button 
          className="blend-button"
          onClick={handleBlend}
          disabled={!sequence1 || !sequence2}
        >
          Proceed to Blend
        </button>
      </div>
    </div>
  )
}