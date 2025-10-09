/**
 * SequenceViewer - Motion Capture Sequence Display Component
 * 
 * This component provides a detailed view of a single motion capture sequence
 * with timeline scrubbing, metadata display, and frame-by-frame navigation.
 * 
 * Key Features:
 * - Motion sequence metadata display (name, joint count, frame count)
 * - Timeline scrubber for frame-accurate navigation
 * - Skeleton visualization placeholder (ready for 3D viewer integration)
 * - Real-time frame information display
 * - Active/inactive visual states for dual-viewer scenarios
 * 
 * OBS-Style Design:
 * - Professional header with sequence info
 * - Large preview area for motion visualization
 * - Integrated timeline controls at bottom
 */
import React, {useState, useEffect} from 'react'
import Timeline from './Timeline'

/**
 * Motion sequence data structure
 * Normalized format for displaying motion capture data
 */
interface SequenceData {
  name: string          // Display name of the sequence
  shape: number[]       // Array dimensions [frames, joints, coordinates]
  frames: number        // Total number of animation frames
  joints: number        // Number of skeleton joints/markers
  duration?: number     // Sequence duration in seconds (optional)
  data?: any           // Raw motion data for processing
}

/**
 * SequenceViewer component props
 * Designed for dual-viewer synchronization
 */
interface SequenceViewerProps {
  /** Motion sequence data to display */
  sequence: SequenceData | null
  /** Current frame position (0-based index) */
  currentFrame: number
  /** Callback when user changes frame position */
  onFrameChange: (frame: number) => void
  /** Display title for this viewer */
  title: string
  /** Whether this viewer is currently active/focused */
  isActive?: boolean
}

/**
 * SequenceViewer main component implementation
 * 
 * Renders either an empty state (when no sequence loaded) or a full
 * sequence viewer with metadata, preview area, and timeline controls.
 */
export default function SequenceViewer({sequence, currentFrame, onFrameChange, title, isActive}: SequenceViewerProps) {
  // Handle empty state - show placeholder with drop zone styling
  if (!sequence) {
    return (
      <div className="sequence-viewer empty">
        <div className="viewer-header">
          <h3>{title}</h3>
          <div className="status">No sequence loaded</div>
        </div>
        <div className="viewer-content">
          {/* Future: Add drag-and-drop functionality for motion files */}
          <div className="empty-state">Drop a motion file here</div>
        </div>
      </div>
    )
  }

  // Render full sequence viewer with all features
  return (
    <div className={`sequence-viewer ${isActive ? 'active' : ''}`}>
      {/* Header with sequence metadata - OBS-style info panel */}
      <div className="viewer-header">
        <h3>{title}</h3>
        <div className="sequence-info">
          <span className="name">{sequence.name}</span>                    {/* Motion sequence name */}
          <span className="joints">{sequence.joints} joints</span>         {/* Number of skeleton joints */}
          <span className="frames">{sequence.frames} frames</span>         {/* Total frame count */}
          {sequence.duration && (
            <span className="duration">{sequence.duration.toFixed(2)}s</span> /* Sequence duration */
          )}
        </div>
      </div>
      
      {/* Main preview area - ready for 3D skeleton visualization */}
      <div className="viewer-content">
        <div className="mocap-preview">
          {/* Real-time frame information display */}
          <div className="frame-info">
            Frame {currentFrame + 1} / {sequence.frames}
            {sequence.duration && (
              <span className="time-code">
                ({((currentFrame / sequence.frames) * sequence.duration).toFixed(2)}s)
              </span>
            )}
          </div>
          
          {/* Skeleton visualization area */}
          <div className="skeleton-view">
            {/* TODO: Replace with actual 3D skeleton renderer */}
            {/* Future: Integrate Three.js or Canvas-based mocap visualization */}
            <div className="skeleton-placeholder">
              <div className="joint-count">{sequence.joints} joints</div>
              <div className="frame-indicator">Frame {currentFrame + 1}</div>
              <div className="visualization-note">
                🦴 Skeleton visualization ready for implementation
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Integrated timeline control - synchronized across viewers */}
      <Timeline 
        frames={sequence.frames}
        currentFrame={currentFrame}
        onFrameChange={onFrameChange}
      />
    </div>
  )
}