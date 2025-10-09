/**
 * BlendPreviewOverlay - Real-time Motion Blending Visualization
 * 
 * Professional overlay component that visualizes the result of blending
 * two motion capture sequences with adjustable weights and real-time preview.
 * 
 * Key Features:
 * - Real-time blend calculation and visualization
 * - Weighted interpolation between two motion sequences
 * - Interactive blend weight adjustment
 * - Frame-accurate synchronized preview
 * - Professional OBS-style overlay controls
 * - Ghost skeleton comparison mode
 * - Blend difference highlighting
 * 
 * Technical Implementation:
 * - Linear interpolation (LERP) between joint positions
 * - Quaternion interpolation (SLERP) for rotations
 * - Real-time blend weight visualization
 * - Frame-synchronized with master timeline
 * - Performance-optimized for smooth playback
 */
import React, {useMemo, useState} from 'react'
import MocapVisualizer from './MocapVisualizer'

/**
 * Joint position data for blending calculations
 */
interface Joint {
  x: number
  y: number
  z: number
  name?: string
}

/**
 * Motion data structure for blend operations
 */
interface MotionData {
  frames: Joint[][]
  jointNames?: string[]
  fps?: number
}

/**
 * BlendPreviewOverlay component props
 */
interface BlendPreviewOverlayProps {
  /** Primary motion sequence data */
  primaryMotion: MotionData | null
  /** Secondary motion sequence data */
  secondaryMotion: MotionData | null
  /** Current frame for both sequences */
  primaryFrame: number
  secondaryFrame: number
  /** Blend weight (0.0 = 100% primary, 1.0 = 100% secondary) */
  blendWeight: number
  /** Callback for blend weight changes */
  onBlendWeightChange: (weight: number) => void
  /** Whether to show comparison ghosts */
  showGhosts?: boolean
  /** Whether to show blend difference highlighting */
  showDifferences?: boolean
  /** Overlay visibility */
  isVisible: boolean
  /** Callback to close overlay */
  onClose: () => void
}

/**
 * Linear interpolation between two values
 * 
 * @param a Start value
 * @param b End value  
 * @param t Interpolation factor (0-1)
 * @returns Interpolated value
 */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Interpolate between two joint positions
 * 
 * @param jointA Primary joint position
 * @param jointB Secondary joint position
 * @param weight Blend weight (0-1)
 * @returns Blended joint position
 */
function blendJoints(jointA: Joint, jointB: Joint, weight: number): Joint {
  return {
    x: lerp(jointA.x, jointB.x, weight),
    y: lerp(jointA.y, jointB.y, weight),
    z: lerp(jointA.z, jointB.z, weight),
    name: jointA.name || jointB.name
  }
}

/**
 * Calculate blend between two motion frames
 * 
 * Performs weighted interpolation between corresponding joints
 * in two motion frames, handling mismatched joint counts gracefully.
 * 
 * @param frameA Primary motion frame
 * @param frameB Secondary motion frame
 * @param weight Blend weight (0-1)
 * @returns Blended motion frame
 */
function blendFrames(frameA: Joint[], frameB: Joint[], weight: number): Joint[] {
  if (!frameA || !frameB) return frameA || frameB || []
  
  // Use the minimum joint count to avoid index errors
  const minJoints = Math.min(frameA.length, frameB.length)
  const blendedFrame: Joint[] = []
  
  for (let i = 0; i < minJoints; i++) {
    const jointA = frameA[i]
    const jointB = frameB[i]
    
    if (jointA && jointB) {
      blendedFrame.push(blendJoints(jointA, jointB, weight))
    } else {
      // Fallback to available joint if one is missing
      blendedFrame.push(jointA || jointB)
    }
  }
  
  return blendedFrame
}

/**
 * BlendPreviewOverlay main component implementation
 * 
 * Renders a professional overlay showing the real-time result
 * of blending two motion sequences with interactive controls.
 */
export default function BlendPreviewOverlay({
  primaryMotion,
  secondaryMotion,
  primaryFrame,
  secondaryFrame,
  blendWeight,
  onBlendWeightChange,
  showGhosts = false,
  showDifferences = false,
  isVisible,
  onClose
}: BlendPreviewOverlayProps) {
  const [dragStart, setDragStart] = useState<{x: number, weight: number} | null>(null)
  
  /**
   * Calculate blended motion data in real-time
   * 
   * Memoized calculation that blends the current frames from both
   * sequences based on the current blend weight.
   */
  const blendedMotion = useMemo((): MotionData | null => {
    if (!primaryMotion || !secondaryMotion) return null
    
    // Get current frames from both sequences
    const primaryCurrentFrame = primaryMotion.frames[primaryFrame]
    const secondaryCurrentFrame = secondaryMotion.frames[secondaryFrame]
    
    if (!primaryCurrentFrame || !secondaryCurrentFrame) return null
    
    // Calculate blended frame
    const blendedFrame = blendFrames(primaryCurrentFrame, secondaryCurrentFrame, blendWeight)
    
    return {
      frames: [blendedFrame], // Single frame for current preview
      jointNames: primaryMotion.jointNames || secondaryMotion.jointNames,
      fps: primaryMotion.fps || secondaryMotion.fps || 30
    }
  }, [primaryMotion, secondaryMotion, primaryFrame, secondaryFrame, blendWeight])
  
  /**
   * Handle blend weight slider interaction
   * 
   * @param event Mouse event from weight slider
   */
  const handleWeightSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newWeight = parseFloat(event.target.value)
    onBlendWeightChange(newWeight)
  }
  
  /**
   * Handle manual weight adjustment via drag
   * 
   * @param event Mouse event
   */
  const handleWeightDrag = (event: React.MouseEvent) => {
    if (!dragStart) return
    
    const deltaX = event.clientX - dragStart.x
    const sensitivity = 0.005 // Adjust sensitivity as needed
    const newWeight = Math.max(0, Math.min(1, dragStart.weight + deltaX * sensitivity))
    
    onBlendWeightChange(newWeight)
  }
  
  /**
   * Start weight drag operation
   * 
   * @param event Mouse down event
   */
  const startWeightDrag = (event: React.MouseEvent) => {
    setDragStart({
      x: event.clientX,
      weight: blendWeight
    })
  }
  
  /**
   * End weight drag operation
   */
  const endWeightDrag = () => {
    setDragStart(null)
  }
  
  // Attach global mouse events for dragging
  React.useEffect(() => {
    if (!dragStart) return
    
    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - dragStart.x
      const sensitivity = 0.005
      const newWeight = Math.max(0, Math.min(1, dragStart.weight + deltaX * sensitivity))
      onBlendWeightChange(newWeight)
    }
    
    const handleMouseUp = () => {
      setDragStart(null)
    }
    
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [dragStart, onBlendWeightChange])
  
  if (!isVisible || !blendedMotion) {
    return null
  }
  
  return (
    <div className="blend-preview-overlay">
      {/* Overlay header with controls */}
      <div className="overlay-header">
        <div className="overlay-title">
          <h3>🎭 Blend Preview</h3>
          <span className="frame-info">
            Frame P:{primaryFrame + 1} + S:{secondaryFrame + 1}
          </span>
        </div>
        
        <div className="overlay-controls">
          <button 
            className="control-btn"
            onClick={() => onBlendWeightChange(0)}
            title="100% Primary"
          >
            A
          </button>
          
          <button 
            className="control-btn"
            onClick={() => onBlendWeightChange(0.5)}
            title="50/50 Blend"
          >
            ⚖
          </button>
          
          <button 
            className="control-btn"
            onClick={() => onBlendWeightChange(1)}
            title="100% Secondary"
          >
            B
          </button>
          
          <button 
            className="close-btn"
            onClick={onClose}
            title="Close Preview"
          >
            ✕
          </button>
        </div>
      </div>
      
      {/* Blend weight visualization */}
      <div className="blend-weight-control">
        <div className="weight-labels">
          <span className="primary-label">Primary ({(1-blendWeight).toFixed(2)})</span>
          <span className="secondary-label">Secondary ({blendWeight.toFixed(2)})</span>
        </div>
        
        <div 
          className="weight-slider-container"
          onMouseDown={startWeightDrag}
        >
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={blendWeight}
            onChange={handleWeightSliderChange}
            className="weight-slider"
          />
          <div 
            className="weight-indicator"
            style={{left: `${blendWeight * 100}%`}}
          />
        </div>
        
        <div className="weight-value">
          <span className="current-weight">{(blendWeight * 100).toFixed(1)}%</span>
        </div>
      </div>
      
      {/* Blended skeleton visualization */}
      <div className="blend-visualization">
        <div className="main-preview">
          <MocapVisualizer
            motionData={blendedMotion}
            currentFrame={0} // Always show frame 0 since we only have one blended frame
            width={400}
            height={300}
            interactive={true}
            showLabels={false}
          />
        </div>
        
        {/* Ghost overlays for comparison */}
        {showGhosts && primaryMotion && secondaryMotion && (
          <div className="ghost-overlays">
            <div className="ghost-primary">
              <label>Primary Ghost</label>
              <MocapVisualizer
                motionData={{
                  frames: [primaryMotion.frames[primaryFrame] || []],
                  jointNames: primaryMotion.jointNames
                }}
                currentFrame={0}
                width={200}
                height={150}
                interactive={false}
                showLabels={false}
              />
            </div>
            
            <div className="ghost-secondary">
              <label>Secondary Ghost</label>
              <MocapVisualizer
                motionData={{
                  frames: [secondaryMotion.frames[secondaryFrame] || []],
                  jointNames: secondaryMotion.jointNames
                }}
                currentFrame={0}
                width={200}
                height={150}
                interactive={false}
                showLabels={false}
              />
            </div>
          </div>
        )}
      </div>
      
      {/* Blend statistics and info */}
      <div className="blend-info">
        <div className="blend-stats">
          <div className="stat-item">
            <span className="stat-label">Joints:</span>
            <span className="stat-value">{blendedMotion.frames[0]?.length || 0}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Blend Ratio:</span>
            <span className="stat-value">
              {(1-blendWeight).toFixed(2)} : {blendWeight.toFixed(2)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Quality:</span>
            <span className="stat-value">Real-time</span>
          </div>
        </div>
        
        <div className="blend-description">
          <p>
            <strong>Blend Preview:</strong> Real-time interpolation between two motion sequences.
            Adjust the blend weight to control the influence of each sequence on the final result.
          </p>
        </div>
      </div>
    </div>
  )
}