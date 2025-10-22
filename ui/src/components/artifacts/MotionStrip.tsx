/**
 * MotionStrip - Frame Grid with Colored Timeline
 * 
 * Displays a horizontal strip of motion frames with an under-strip timeline
 * showing colored segments, overlaps, and transition windows.
 */

import React, { useMemo, useState, useCallback } from 'react';
import type { MotionStripProps, FrameTooltipData } from '../../types/artifacts';
import UnderStripBand from './UnderStripBand';

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

export default function MotionStrip({
  label,
  thumbnails,
  band,
  fps,
  frameStep,
  highlightWindows = [],
  emphasis,
  onHover,
  metrics
}: MotionStripProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [tooltipData, setTooltipData] = useState<FrameTooltipData | null>(null);
  const total = thumbnails.length;

  const handleHover = useCallback((index: number) => {
    setHoverIdx(index);
    const absoluteFrame = index * frameStep;
    onHover?.(absoluteFrame);

    // Find which segment this frame belongs to
    const segment = band.find(s => absoluteFrame >= s.fromFrame && absoluteFrame <= s.toFrame);
    
    // Extract metrics for this frame if available
    const frameMetrics: FrameTooltipData['metrics'] = {};
    if (metrics && absoluteFrame < metrics.l2Velocity[metrics.joints[0]]?.length) {
      metrics.joints.forEach(joint => {
        const jointKey = joint === 'pelvis' ? 'Hips' : 
                        joint === 'lwrist' ? 'LeftWrist' :
                        joint === 'rwrist' ? 'RightWrist' :
                        joint === 'lfoot' ? 'LeftFoot' :
                        joint === 'rfoot' ? 'RightFoot' : joint;
        
        frameMetrics[joint] = {
          velocity: metrics.l2Velocity[jointKey]?.[absoluteFrame] ?? 0,
          acceleration: metrics.l2Acceleration[jointKey]?.[absoluteFrame] ?? 0
        };
      });
    }

    setTooltipData({
      frameIndex: absoluteFrame,
      timeSeconds: absoluteFrame / fps,
      segmentLabel: segment?.label ?? 'Unknown',
      metrics: Object.keys(frameMetrics).length > 0 ? frameMetrics : undefined
    });
  }, [frameStep, onHover, band, fps, metrics]);

  const handleLeave = useCallback(() => {
    setHoverIdx(null);
    setTooltipData(null);
  }, []);

  return (
    <div className="motion-strip">
      {/* Header */}
      <div className="strip-header">
        <h4 className={cn('strip-label', emphasis && 'strip-label-emphasis')}>
          {label}
        </h4>
        {tooltipData && (
          <div className="strip-tooltip">
            <div className="tooltip-frame">
              Frame: #{tooltipData.frameIndex} ({tooltipData.timeSeconds.toFixed(2)}s)
            </div>
            <div className="tooltip-segment">
              Segment: {tooltipData.segmentLabel}
            </div>
            {tooltipData.metrics && (
              <div className="tooltip-metrics">
                {Object.entries(tooltipData.metrics).map(([joint, data]) => (
                  <div key={joint} className="tooltip-metric-row">
                    <span className="tooltip-joint">{joint}:</span>
                    <span className="tooltip-values">
                      v: {data.velocity.toFixed(3)} | a: {data.acceleration.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Frame grid */}
      <div className={cn(
        'frame-grid',
        emphasis && 'frame-grid-emphasis'
      )}>
        {thumbnails.map((src, i) => (
          <button
            key={i}
            className={cn(
              'frame-tile',
              hoverIdx === i && 'frame-tile-hover'
            )}
            onMouseEnter={() => handleHover(i)}
            onMouseLeave={handleLeave}
            aria-label={`Frame ${i * frameStep}`}
            tabIndex={0}
          >
            {/* Placeholder since we don't have actual thumbnails yet */}
            <div className="frame-placeholder">
              <span className="frame-number">{i * frameStep}</span>
            </div>
          </button>
        ))}
      </div>

      {/* Under-strip band */}
      <UnderStripBand
        totalSamples={total}
        segments={band}
        frameStep={frameStep}
        highlightWindows={highlightWindows}
      />
    </div>
  );
}
