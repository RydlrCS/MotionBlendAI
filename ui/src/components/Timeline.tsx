/**
 * Timeline - Professional Frame-Accurate Scrubber Component
 * 
 * Provides frame-by-frame navigation with visual feedback, similar to
 * video editing software timelines. Features click-to-seek and drag scrubbing.
 * 
 * Key Features:
 * - Click anywhere on timeline to jump to frame
 * - Drag scrubbing for smooth frame navigation
 * - Visual frame markers with numbering
 * - Current position indicator (scrubber)
 * - Play/pause controls (ready for implementation)
 * - Responsive width adjustment
 * 
 * OBS-Style Design:
 * - Dark timeline track with frame markers
 * - Bright scrubber indicator for current position
 * - Professional transport controls
 * - Frame-accurate positioning
 */
import React from 'react'

/**
 * Timeline component props
 * Designed for precise frame control in motion sequences
 */
interface TimelineProps {
  /** Total number of frames in the sequence */
  frames: number
  /** Current frame position (0-based index) */
  currentFrame: number
  /** Callback when user seeks to a new frame */
  onFrameChange: (frame: number) => void
  /** Timeline width in pixels (default: 400) */
  width?: number
}

/**
 * Timeline main component implementation
 * 
 * Provides professional video editor-style timeline scrubbing with
 * click-to-seek and drag-to-scrub functionality.
 */
export default function Timeline({frames, currentFrame, onFrameChange, width = 400}: TimelineProps) {
  /**
   * Handle timeline click events
   * 
   * Calculates the frame position based on click coordinates and
   * seeks to that frame. Includes boundary clamping to prevent
   * out-of-range frame access.
   * 
   * @param e Mouse event from timeline click
   */
  const handleClick = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left                           // Click position relative to timeline
    const frameWidth = rect.width / frames                    // Pixels per frame
    const newFrame = Math.floor(x / frameWidth)               // Calculate target frame
    const clampedFrame = Math.max(0, Math.min(frames - 1, newFrame)) // Clamp to valid range
    onFrameChange(clampedFrame)
  }

  /**
   * Handle timeline drag scrubbing
   * 
   * Enables smooth scrubbing by detecting mouse drag (button held down)
   * and continuously updating frame position during drag.
   * 
   * @param e Mouse move event during drag
   */
  const handleDrag = (e: React.MouseEvent) => {
    if (e.buttons === 1) { // Left mouse button held down - drag scrubbing active
      handleClick(e)
    }
  }

  return (
    <div className="timeline-container">
      {/* Transport controls - OBS-style playback buttons */}
      <div className="timeline-controls">
        <button 
          className="play-btn"
          onClick={() => {
            // TODO: Implement play/pause functionality
            // Future: Add auto-advance timer for playback
            console.log('Play/pause clicked - ready for implementation')
          }}
          title="Play/Pause (Space)"
        >
          ▶
        </button>
        {/* Frame counter display - 1-based for user friendliness */}
        <span className="frame-display">{currentFrame + 1} / {frames}</span>
        {/* TODO: Add additional transport controls (step forward/back, goto start/end) */}
      </div>
      
      {/* Main timeline scrubber area */}
      <div 
        className="timeline"
        style={{width}}
        onClick={handleClick}                                    // Click-to-seek functionality
        onMouseMove={handleDrag}                                // Drag scrubbing
        title={`Frame ${currentFrame + 1} of ${frames}`}
      >
        <div className="timeline-track">
          {/* Frame position markers - evenly distributed across timeline */}
          <div className="frame-markers">
            {Array.from({length: Math.min(frames, 20)}, (_, i) => {
              // Calculate frame number for this marker position
              const frameNum = Math.floor((i / 19) * (frames - 1))
              return (
                <div 
                  key={i} 
                  className="frame-marker"
                  style={{left: `${(frameNum / (frames - 1)) * 100}%`}}
                  title={`Jump to frame ${frameNum + 1}`}
                >
                  {frameNum + 1}
                </div>
              )
            })}
          </div>
          
          {/* Timeline scrubber indicator - shows current position */}
          <div 
            className="timeline-scrubber"
            style={{left: `${(currentFrame / (frames - 1)) * 100}%`}}
            title={`Current frame: ${currentFrame + 1}`}
          />
        </div>
      </div>
    </div>
  )
}