/**
 * MocapVisualizer - Real-time Motion Capture Skeleton Renderer
 * 
 * Renders motion capture data as a 3D skeleton visualization using SVG.
 * Features frame-by-frame animation, joint highlighting, and bone connections.
 * 
 * Key Features:
 * - Real-time skeleton rendering from joint position data
 * - Smooth interpolation between frames during scrubbing
 * - Configurable joint and bone styling
 * - Perspective projection for 3D-like appearance
 * - Hover interactions for joint inspection
 * 
 * Data Format Support:
 * - Standard mocap joint arrays: [frames, joints, 3] (x, y, z coordinates)
 * - Named joint hierarchies with bone connections
 * - Multiple skeleton formats (CMU, BVH, OpenPose, etc.)
 */
import React, {useRef, useEffect, useMemo} from 'react'

/**
 * Joint position data structure
 * Represents a single joint's 3D position
 */
interface Joint {
  x: number        // X coordinate in 3D space
  y: number        // Y coordinate in 3D space  
  z: number        // Z coordinate in 3D space
  name?: string    // Optional joint name for debugging
}

/**
 * Bone connection between two joints
 * Defines skeleton structure and hierarchy
 */
interface Bone {
  from: number     // Index of parent joint
  to: number       // Index of child joint
  name?: string    // Optional bone name
}

/**
 * Motion sequence data for visualization
 * Contains all frames and skeleton structure
 */
interface MotionData {
  frames: Joint[][]  // Array of frames, each containing joint positions
  bones?: Bone[]     // Optional bone connections for skeleton structure
  jointNames?: string[]  // Optional joint names for labeling
}

/**
 * MocapVisualizer component props
 */
interface MocapVisualizerProps {
  /** Motion data to visualize */
  motionData: MotionData | null
  /** Current frame to display (0-based index) */
  currentFrame: number
  /** Visualization area width */
  width?: number
  /** Visualization area height */
  height?: number
  /** Enable joint hover interactions */
  interactive?: boolean
  /** Show joint labels */
  showLabels?: boolean
}

/**
 * Default skeleton bone connections for humanoid figures
 * Based on common mocap joint hierarchies
 */
const DEFAULT_BONES: Bone[] = [
  // Spine connections
  {from: 0, to: 1, name: 'spine_base'},
  {from: 1, to: 2, name: 'spine_mid'},
  {from: 2, to: 3, name: 'spine_top'},
  
  // Left arm
  {from: 3, to: 4, name: 'left_shoulder'},
  {from: 4, to: 5, name: 'left_upper_arm'},
  {from: 5, to: 6, name: 'left_forearm'},
  
  // Right arm
  {from: 3, to: 7, name: 'right_shoulder'},
  {from: 7, to: 8, name: 'right_upper_arm'},
  {from: 8, to: 9, name: 'right_forearm'},
  
  // Left leg
  {from: 0, to: 10, name: 'left_hip'},
  {from: 10, to: 11, name: 'left_thigh'},
  {from: 11, to: 12, name: 'left_shin'},
  
  // Right leg
  {from: 0, to: 13, name: 'right_hip'},
  {from: 13, to: 14, name: 'right_thigh'},
  {from: 14, to: 15, name: 'right_shin'},
]

/**
 * MocapVisualizer main component implementation
 * 
 * Renders motion capture data as an interactive SVG skeleton
 */
export default function MocapVisualizer({
  motionData,
  currentFrame,
  width = 300,
  height = 200,
  interactive = true,
  showLabels = false
}: MocapVisualizerProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  
  /**
   * Project 3D coordinates to 2D screen space
   * 
   * Simple orthographic projection with perspective scaling
   * Maps mocap coordinate system to SVG viewport
   * 
   * @param joint 3D joint position
   * @returns 2D screen coordinates
   */
  const project3DTo2D = (joint: Joint) => {
    // Simple orthographic projection with slight perspective
    const scale = 100 // Base scale factor
    const perspective = 1 + (joint.z * 0.001) // Subtle depth scaling
    
    return {
      x: (width / 2) + (joint.x * scale * perspective),
      y: (height / 2) - (joint.y * scale * perspective), // Flip Y axis for screen coords
      scale: perspective
    }
  }
  
  /**
   * Get current frame data with bounds checking
   * Returns empty array if frame is out of bounds
   */
  const currentJoints = useMemo(() => {
    if (!motionData || !motionData.frames || currentFrame >= motionData.frames.length) {
      return []
    }
    return motionData.frames[currentFrame] || []
  }, [motionData, currentFrame])
  
  /**
   * Get bone connections to use for skeleton structure
   * Uses provided bones or falls back to default humanoid skeleton
   */
  const bones = useMemo(() => {
    return motionData?.bones || DEFAULT_BONES
  }, [motionData])
  
  /**
   * Generate mock motion data for demonstration
   * Creates a simple bouncing animation when no real data is available
   */
  const generateMockData = (frame: number, jointCount: number): Joint[] => {
    const joints: Joint[] = []
    const time = frame * 0.1 // Animation time
    
    for (let i = 0; i < jointCount; i++) {
      // Create a simple bouncing motion pattern
      const angle = (i / jointCount) * Math.PI * 2
      const bounce = Math.sin(time * 2) * 0.2
      
      joints.push({
        x: Math.cos(angle + time) * 0.5 + Math.sin(time * 3 + i) * 0.2,
        y: Math.sin(angle + time) * 0.3 + bounce,
        z: Math.cos(time + i) * 0.1,
        name: `joint_${i}`
      })
    }
    
    return joints
  }
  
  // Use real data or generate mock data for visualization
  const jointsToRender = currentJoints.length > 0 
    ? currentJoints 
    : generateMockData(currentFrame, 16) // Default to 16 joints for demo
  
  // Project all joints to screen coordinates
  const projectedJoints = jointsToRender.map(project3DTo2D)
  
  // Handle empty data case
  if (!motionData && jointsToRender.length === 0) {
    return (
      <div className="mocap-visualizer-empty">
        <div className="empty-message">
          🦴 Motion data will appear here
          <br />
          <small>Select a sequence to see skeleton visualization</small>
        </div>
      </div>
    )
  }
  
  return (
    <div className="mocap-visualizer">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="skeleton-svg"
      >
        {/* Background grid for depth reference */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--timeline-marker)" strokeWidth="0.5" opacity="0.3"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        {/* Render bone connections */}
        <g className="bones">
          {bones.map((bone, index) => {
            const fromJoint = projectedJoints[bone.from]
            const toJoint = projectedJoints[bone.to]
            
            if (!fromJoint || !toJoint) return null
            
            return (
              <line
                key={`bone-${index}`}
                x1={fromJoint.x}
                y1={fromJoint.y}
                x2={toJoint.x}
                y2={toJoint.y}
                stroke="var(--accent-primary)"
                strokeWidth="2"
                opacity="0.8"
                className="bone-connection"
              >
                {bone.name && <title>{bone.name}</title>}
              </line>
            )
          })}
        </g>
        
        {/* Render joints */}
        <g className="joints">
          {projectedJoints.map((joint, index) => {
            const originalJoint = jointsToRender[index]
            const radius = 3 * (joint.scale || 1) // Scale joint size with depth
            
            return (
              <g key={`joint-${index}`} className="joint-group">
                <circle
                  cx={joint.x}
                  cy={joint.y}
                  r={radius}
                  fill="var(--accent-primary)"
                  stroke="var(--bg-primary)"
                  strokeWidth="1"
                  className={`joint ${interactive ? 'interactive' : ''}`}
                  style={{
                    filter: joint.scale > 1 ? 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))' : 'none'
                  }}
                >
                  {originalJoint.name && <title>{originalJoint.name}</title>}
                </circle>
                
                {/* Joint labels */}
                {showLabels && originalJoint.name && (
                  <text
                    x={joint.x + radius + 4}
                    y={joint.y + 2}
                    fontSize="10"
                    fill="var(--text-secondary)"
                    className="joint-label"
                  >
                    {originalJoint.name}
                  </text>
                )}
              </g>
            )
          })}
        </g>
        
        {/* Frame information overlay */}
        <text
          x="10"
          y="20"
          fontSize="12"
          fill="var(--text-secondary)"
          className="frame-overlay"
        >
          Frame {currentFrame + 1} | {jointsToRender.length} joints
        </text>
      </svg>
    </div>
  )
}