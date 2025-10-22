/**
 * UnderStripBand - Colored timeline visualization under frame grid
 * Shows segments with color-coded bands and transition windows
 */
import React from 'react';
import type { UnderStripBandProps, Segment } from '../types/artifacts';

interface SegmentWithLayout extends Segment {
  widthPct: number;
  leftPct: number;
}

export default function UnderStripBand({
  totalSamples,
  segments,
  frameStep,
  highlightWindows = []
}: UnderStripBandProps) {
  const totalFrames = totalSamples * frameStep;
  
  // Calculate segment positions as percentages
  const segmentsWithLayout: SegmentWithLayout[] = segments.map(s => {
    const width = ((s.toFrame - s.fromFrame + 1) / totalFrames) * 100;
    const left = (s.fromFrame / totalFrames) * 100;
    return {
      ...s,
      widthPct: width,
      leftPct: left
    };
  });
  
  return (
    <div className="relative h-3 mt-2 rounded-full bg-gray-200 overflow-hidden">
      {/* Render segments */}
      {segmentsWithLayout.map((segment, idx) => (
        <span
          key={`segment-${idx}`}
          title={segment.label}
          style={{
            left: `${segment.leftPct}%`,
            width: `${segment.widthPct}%`,
            backgroundColor: segment.color,
            opacity: segment.alpha ?? 1
          }}
          className="absolute h-full transition-opacity hover:opacity-100"
          aria-label={`${segment.label} from frame ${segment.fromFrame} to ${segment.toFrame}`}
        />
      ))}
      
      {/* Render transition windows as dashed borders */}
      {highlightWindows.map((window, idx) => {
        const left = (window.start / totalFrames) * 100;
        const width = ((window.end - window.start) / totalFrames) * 100;
        
        return (
          <span
            key={`window-${idx}`}
            style={{
              left: `${left}%`,
              width: `${width}%`
            }}
            className="absolute h-full border-2 border-dashed border-yellow-500 rounded-full pointer-events-none"
            aria-label={`Transition window: frames ${window.start} to ${window.end}`}
          />
        );
      })}
    </div>
  );
}
