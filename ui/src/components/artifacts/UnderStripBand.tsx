/**
 * UnderStripBand - Colored Timeline with Segments
 * 
 * Displays a colored band underneath the frame grid showing motion segments,
 * overlaps, and transition windows with dashed markers.
 */

import React, { useMemo } from 'react';
import type { UnderStripBandProps, Segment } from '../../types/artifacts';

type ComputedSegment = Segment & {
  widthPct: number;
  leftPct: number;
}

export default function UnderStripBand({
  totalSamples,
  segments,
  frameStep,
  highlightWindows = []
}: UnderStripBandProps) {
  const computedSegments = useMemo(() => {
    const totalFrames = totalSamples * frameStep;
    return segments.map(s => {
      const w = ((s.toFrame - s.fromFrame + 1) / totalFrames) * 100;
      const l = (s.fromFrame / totalFrames) * 100;
      return { ...s, widthPct: w, leftPct: l };
    });
  }, [segments, totalSamples, frameStep]);

  const computedWindows = useMemo(() => {
    const totalFrames = totalSamples * frameStep;
    return highlightWindows.map(w => ({
      leftPct: (w.start / totalFrames) * 100,
      widthPct: ((w.end - w.start) / totalFrames) * 100
    }));
  }, [highlightWindows, totalSamples, frameStep]);

  return (
    <div className="under-strip-band">
      {/* Colored segments */}
      {computedSegments.map((s, k) => (
        <span
          key={k}
          className="segment-span"
          title={`${s.label} (${s.fromFrame}-${s.toFrame})`}
          style={{
            left: `${s.leftPct}%`,
            width: `${s.widthPct}%`,
            backgroundColor: s.color,
            opacity: s.alpha ?? 1
          }}
        />
      ))}
      
      {/* Transition window markers */}
      {computedWindows.map((w, k) => (
        <span
          key={`window-${k}`}
          className="transition-window"
          style={{
            left: `${w.leftPct}%`,
            width: `${w.widthPct}%`
          }}
        />
      ))}
    </div>
  );
}
