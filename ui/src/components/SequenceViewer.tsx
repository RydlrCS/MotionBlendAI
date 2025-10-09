/**
 * SequenceViewer - Motion Capture Sequence Display Component
 * 
 * This component provides a detailed view of a single motion capture sequence
 * with timeline scrubbing, metadata display, and real-time skeleton visualization.
 * 
 * Key Features:
 * - Motion sequence metadata display (name, joint count, frame count)
 * - Timeline scrubber for frame-accurate navigation
 * - Real-time skeleton visualization with MocapVisualizer
 * - Frame-by-frame motion preview
 * - Active/inactive visual states for dual-viewer scenarios
 * 
 * OBS-Style Design:
 * - Professional header with sequence info
 * - Large preview area with skeleton rendering
 * - Integrated timeline controls at bottom
 */
import React, {useState, useEffect} from 'react'
import Timeline from './Timeline'
import MocapVisualizer from './MocapVisualizer'

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
  /** Sequence data to display, or null for empty state */
  sequence: SequenceData | null
  /** Whether this viewer is the active/primary viewer */
  isActive: boolean
  /** Current frame position for timeline sync */
  currentFrame: number
  /** Callback when frame position changes */
  onFrameChange: (frame: number) => void
  /** Whether playback is currently active */
  isPlaying?: boolean
}

/**
 * Generate mock motion data for visualization
 * 
 * Creates realistic skeletal motion for demonstration purposes.
 * In production, this would be replaced with actual motion data
 * extracted from GLB, TRC, FBX, or NPY files.
 * 
 * @param sequence Sequence metadata
 * @returns Mock motion data for MocapVisualizer
 */
function generateMockMotionData(sequence: SequenceData) {
  const jointCount = sequence.joints || 22 // Default humanoid skeleton
  const frameCount = sequence.frames || 100
  
  // Generate mock skeletal animation (simple walking cycle)
  const frames = []
  for (let frame = 0; frame < frameCount; frame++) {
    const time = (frame / frameCount) * Math.PI * 2 // One full cycle
    const joints = []
    
    for (let joint = 0; joint < jointCount; joint++) {
      // Generate realistic joint positions with simple animation
      const baseY = joint < 4 ? 1.0 + joint * 0.3 : 0.5 // Spine higher
      const animOffset = Math.sin(time + joint * 0.1) * 0.1
      
      joints.push({
        x: Math.sin(time * 0.5 + joint) * 0.3, // X - gentle sway
        y: baseY + animOffset,                 // Y - vertical movement  
        z: Math.cos(time + joint * 0.05) * 0.2, // Z - forward/back
        name: [
          'root', 'spine1', 'spine2', 'spine3', 'neck',
          'leftShoulder', 'leftElbow', 'leftWrist', 'leftHand',
          'rightShoulder', 'rightElbow', 'rightWrist', 'rightHand',
          'leftHip', 'leftKnee', 'leftAnkle', 'leftFoot',
          'rightHip', 'rightKnee', 'rightAnkle', 'rightFoot',
          'head'
        ][joint] || `joint_${joint}`
      })
    }
    
    frames.push(joints)
  }
  
  return {
    frames,
    jointNames: [
      'root', 'spine1', 'spine2', 'spine3', 'neck',
      'leftShoulder', 'leftElbow', 'leftWrist', 'leftHand',
      'rightShoulder', 'rightElbow', 'rightWrist', 'rightHand',
      'leftHip', 'leftKnee', 'leftAnkle', 'leftFoot',
      'rightHip', 'rightKnee', 'rightAnkle', 'rightFoot',
      'head'
    ].slice(0, jointCount)
  }
}

/**
 * SequenceViewer main component implementation
 * 
 * Renders either an empty state (when no sequence loaded) or a full
 * sequence viewer with metadata, preview area, and timeline controls.
 */
export default function SequenceViewer({
  sequence, 
  isActive, 
  currentFrame, 
  onFrameChange,
  isPlaying = false
}: SequenceViewerProps) {
  // Handle empty state - show placeholder with drop zone styling
  if (!sequence) {
    return (
      <div className="sequence-viewer empty">
        <div className="viewer-header">
          <h3>Motion Sequence</h3>
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
        <div className="sequence-info">
          <div className="sequence-title">
            {sequence.name}
            {isPlaying && (
              <span className="playback-indicator" title="Synchronized playback active">
                ▶
              </span>
            )}
          </div>
          <div className="sequence-type">Motion Sequence</div>
        </div>
        <div className="sequence-metadata">
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
            {/* Real-time mocap visualization with skeleton rendering */}
            <MocapVisualizer 
              motionData={generateMockMotionData(sequence)}
              currentFrame={currentFrame}
              width={300}
              height={200}
              interactive={true}
              showLabels={false}
            />
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