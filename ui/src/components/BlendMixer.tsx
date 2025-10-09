/**
 * BlendMixer - OBS Studio-style Motion Blending Interface
 * 
 * This component provides a professional video mixer-like interface for blending
 * two motion capture sequences. Features include:
 * - Dual sequence viewers with synchronized timeline scrubbing
 * - Real-time blend weight adjustment
 * - Overlay visualization of blend preview
 * - Motion sequence selection and metadata display
 * - Drag and drop support from ElasticSearch popup
 * - File control panel with search integration
 * 
 * Key Features:
 * - Side-by-side sequence comparison
 * - Frame-accurate timeline synchronization
 * - Visual blend weight slider (0.0 = 100% Sequence A, 1.0 = 100% Sequence B)
 * - Optional blend overlay preview
 * - Professional OBS-style layout and controls
 * - Drag and drop file import from search results
 */
import React, {useState, useEffect} from 'react'
import SequenceViewer from './SequenceViewer'
import FileControlPanel from './FileControlPanel'
import {getMotions} from '../client'
import {useSyncedPlayback} from '../hooks/useSyncedPlayback'
import BlendPreviewOverlay from './BlendPreviewOverlay'

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
 * - selectedFiles: Files selected via search or file panel
 */
export default function BlendMixer({onBlendRequest}: BlendMixerProps) {
  // Motion sequence management
  const [motions, setMotions] = useState<any[]>([])                    // Available motions from API
  const [sequence1, setSequence1] = useState<SequenceData | null>(null) // Selected sequence A
  const [sequence2, setSequence2] = useState<SequenceData | null>(null) // Selected sequence B
  const [selectedFiles, setSelectedFiles] = useState<any[]>([])        // Files from search/panel
  
  // Synchronized playback state for dual viewers
  const syncedPlayback = useSyncedPlayback({
    primaryFrames: sequence1?.frames || 100,
    secondaryFrames: sequence2?.frames || 100,
    fps: 30,
    loop: false
  })
  
  // Blend control state
  const [blendWeight, setBlendWeight] = useState(0.5)                  // Blend weight: 0.0=100% seq1, 1.0=100% seq2
  const [showOverlay, setShowOverlay] = useState(false)                // Show/hide blend preview overlay

  // Load available motions from API
  useEffect(() => {
    getMotions().then(data => {
      setMotions(data.motions || [])
    }).catch(() => setMotions([]))
  }, [])
  
  // Update synchronized playback when sequences change
  useEffect(() => {
    syncedPlayback.updateConfig({
      primaryFrames: sequence1?.frames || 100,
      secondaryFrames: sequence2?.frames || 100
    })
  }, [sequence1, sequence2, syncedPlayback])

  /**
   * Handle drag and drop events for motion files
   */
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }

  const handleDrop = (e: React.DragEvent, target: 'sequence1' | 'sequence2') => {
    e.preventDefault()
    
    try {
      const dragData = JSON.parse(e.dataTransfer.getData('application/json'))
      
      if (dragData.type === 'motion-file') {
        const file = dragData.data
        const sequenceData = createSequenceDataFromFile(file)
        
        if (target === 'sequence1') {
          setSequence1(sequenceData)
        } else {
          setSequence2(sequenceData)
        }
        
        console.log(`Dropped ${file.name} into ${target}`)
        syncedPlayback.stop() // Reset playback when sequence changes
      }
    } catch (error) {
      console.error('Failed to process dropped file:', error)
    }
  }

  /**
   * Convert file data from search results to sequence data
   */
  const createSequenceDataFromFile = (file: any): SequenceData => {
    return {
      name: file.name,
      shape: [file.metadata?.frames || 100, file.metadata?.joints || 25, 3],
      frames: file.metadata?.frames || 100,
      joints: file.metadata?.joints || 25,
      duration: file.metadata?.duration || 3.33,
      data: file
    }
  }

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
   * Handle file selection from FileControlPanel
   */
  const handleFileSelect = (file: any) => {
    setSelectedFiles(prev => [...prev, file])
    console.log('File selected from panel:', file.name)
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
  
  /**
   * Handle blend weight adjustment
   * Updates the mixing ratio between primary and secondary sequences
   */
  const handleBlendWeightChange = (weight: number) => {
    setBlendWeight(weight)
    console.log(`Blend weight updated: ${weight.toFixed(2)}`)
  }
  
  /**
   * Convert sequence data to motion data format for blending
   * 
   * @param sequence Sequence data from API
   * @returns Motion data compatible with blend preview
   */
  const generateMotionDataForBlend = (sequence: SequenceData | null) => {
    if (!sequence) return null
    
    // Generate mock motion frames for blending (in production, use real data)
    const frames = []
    for (let frame = 0; frame < sequence.frames; frame++) {
      const joints = []
      const time = (frame / sequence.frames) * Math.PI * 2
      
      for (let joint = 0; joint < sequence.joints; joint++) {
        const baseY = joint < 4 ? 1.0 + joint * 0.3 : 0.5
        const animOffset = Math.sin(time + joint * 0.1) * 0.1
        
        joints.push({
          x: Math.sin(time * 0.5 + joint) * 0.3,
          y: baseY + animOffset,
          z: Math.cos(time + joint * 0.05) * 0.2,
          name: `joint_${joint}`
        })
      }
      
      frames.push(joints)
    }
    
    return {
      frames,
      jointNames: Array.from({length: sequence.joints}, (_, i) => `joint_${i}`),
      fps: 30
    }
  }

  return (
    <div className="blend-mixer">
      <div className="mixer-layout">
        {/* Main mixer area */}
        <div className="mixer-main">
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
                  syncedPlayback.stop() // Reset to beginning when sequence changes
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
                  syncedPlayback.stop() // Reset to beginning when sequence changes
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
            <div 
              className="sequence-drop-zone"
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, 'sequence1')}
            >
              <SequenceViewer 
                sequence={sequence1}
                currentFrame={syncedPlayback.primaryFrame}
                onFrameChange={syncedPlayback.seekTo}
                isActive={true}
                isPlaying={syncedPlayback.isPlaying}
              />
              {!sequence1 && (
                <div className="drop-hint">
                  📎 Drop motion file here or select from dropdown
                </div>
              )}
            </div>
            
            <div 
              className="sequence-drop-zone"
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, 'sequence2')}
            >
              <SequenceViewer 
                sequence={sequence2}
                currentFrame={syncedPlayback.secondaryFrame}
                onFrameChange={syncedPlayback.seekTo}
                isActive={false}
                isPlaying={syncedPlayback.isPlaying}
              />
              {!sequence2 && (
                <div className="drop-hint">
                  📎 Drop motion file here or select from dropdown
                </div>
              )}
            </div>
          </div>

          {/* Blend preview overlay */}
          <BlendPreviewOverlay
            primaryMotion={generateMotionDataForBlend(sequence1)}
            secondaryMotion={generateMotionDataForBlend(sequence2)}
            primaryFrame={syncedPlayback.primaryFrame}
            secondaryFrame={syncedPlayback.secondaryFrame}
            blendWeight={blendWeight}
            onBlendWeightChange={handleBlendWeightChange}
            showGhosts={false}
            showDifferences={false}
            isVisible={showOverlay && !!sequence1 && !!sequence2}
            onClose={() => setShowOverlay(false)}
          />

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

        {/* Right-side file control panel */}
        <div className="mixer-sidebar">
          <FileControlPanel 
            onFileSelect={handleFileSelect}
            selectedFiles={selectedFiles}
          />
        </div>
      </div>
    </div>
  )
}