/**
 * FrameTooltip - Enhanced tooltip showing comprehensive frame metrics
 * Displays frame info, segment, and all joint velocity/acceleration data
 */
import React, { useEffect, useRef, useState } from 'react';
import type { FrameTooltipData, ArtifactMetrics, Segment } from '../types/artifacts';
import '../styles/FrameTooltip.css';

export interface FrameTooltipProps {
  frameIndex: number;
  frameStep: number;
  fps: number;
  segment?: Segment | null;
  metrics?: ArtifactMetrics | null;
  position: { x: number; y: number };
  visible: boolean;
}

const JOINT_DISPLAY_NAMES: Record<string, string> = {
  'pelvis': 'Pelvis',
  'lwrist': 'L Wrist',
  'rwrist': 'R Wrist',
  'lfoot': 'L Foot',
  'rfoot': 'R Foot',
};

export default function FrameTooltip({
  frameIndex,
  frameStep,
  fps,
  segment,
  metrics,
  position,
  visible
}: FrameTooltipProps) {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [adjustedPosition, setAdjustedPosition] = useState(position);

  // Collision detection and viewport bounds adjustment
  useEffect(() => {
    if (!visible || !tooltipRef.current) return;

    const tooltip = tooltipRef.current;
    const rect = tooltip.getBoundingClientRect();
    const viewport = {
      width: window.innerWidth,
      height: window.innerHeight,
    };

    let { x, y } = position;

    // Adjust horizontal position if tooltip goes off-screen
    if (x + rect.width > viewport.width - 10) {
      x = viewport.width - rect.width - 10;
    }
    if (x < 10) {
      x = 10;
    }

    // Adjust vertical position if tooltip goes off-screen
    // Default: show above the cursor
    let finalY = y - rect.height - 10;
    
    if (finalY < 10) {
      // If doesn't fit above, show below
      finalY = y + 20;
    }
    
    if (finalY + rect.height > viewport.height - 10) {
      // If doesn't fit below either, show at top
      finalY = 10;
    }

    setAdjustedPosition({ x, y: finalY });
  }, [position, visible]);

  if (!visible) return null;

  const absoluteFrame = frameIndex * frameStep;
  const timestamp = (absoluteFrame / fps).toFixed(2);

  // Get joint metrics
  const jointMetrics: Array<{ joint: string; v: number; a: number }> = [];
  if (metrics) {
    metrics.joints.forEach(joint => {
      const jointKey = joint === 'pelvis' ? 'Hips' : joint;
      const velocities = metrics.l2Velocity[jointKey] || [];
      const accelerations = metrics.l2Acceleration[jointKey] || [];
      
      if (absoluteFrame < velocities.length) {
        jointMetrics.push({
          joint: JOINT_DISPLAY_NAMES[joint] || joint,
          v: velocities[absoluteFrame],
          a: accelerations[absoluteFrame]
        });
      }
    });
  }

  // Get max values for relative bar visualization
  const maxVelocity = Math.max(...jointMetrics.map(j => j.v), 0.001);
  const maxAcceleration = Math.max(...jointMetrics.map(j => j.a), 0.001);

  return (
    <div
      ref={tooltipRef}
      className="frame-tooltip-enhanced"
      style={{
        position: 'fixed',
        left: `${adjustedPosition.x}px`,
        top: `${adjustedPosition.y}px`,
        transform: 'translateX(-50%)',
        zIndex: 10000,
      }}
      role="tooltip"
      aria-live="polite"
    >
      {/* Header */}
      <div className="tooltip-header">
        <div className="tooltip-frame-info">
          <span className="frame-number">Frame #{absoluteFrame}</span>
          <span className="frame-time">{timestamp}s</span>
        </div>
        {segment && (
          <div className="tooltip-segment">
            <span
              className="segment-color-dot"
              style={{ backgroundColor: segment.color }}
            />
            <span className="segment-label">{segment.label}</span>
          </div>
        )}
      </div>

      {/* Joint Metrics */}
      {jointMetrics.length > 0 && (
        <div className="tooltip-metrics">
          <div className="metrics-header">Joint Metrics</div>
          {jointMetrics.map(({ joint, v, a }) => (
            <div key={joint} className="joint-metric-row">
              <div className="joint-name">{joint}</div>
              <div className="metric-group">
                <div className="metric-item">
                  <span className="metric-label">v:</span>
                  <span className="metric-value">{v.toFixed(4)}</span>
                  <div className="metric-bar">
                    <div
                      className="metric-bar-fill velocity"
                      style={{ width: `${(v / maxVelocity) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="metric-item">
                  <span className="metric-label">a:</span>
                  <span className="metric-value">{a.toFixed(4)}</span>
                  <div className="metric-bar">
                    <div
                      className="metric-bar-fill acceleration"
                      style={{ width: `${(a / maxAcceleration) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Arrow pointer */}
      <div className="tooltip-arrow" />
    </div>
  );
}
