/**
 * MotionStrip - Frame grid visualization with colored timeline band
 * Displays motion thumbnails with hover interaction and metrics display
 */
import React, { useState } from 'react';
import UnderStripBand from './UnderStripBand';
import FrameTooltip from './FrameTooltip';
import type { MotionStripProps } from '../types/artifacts';

export default function MotionStrip({
  label,
  thumbnails,
  band,
  fps,
  frameStep,
  highlightWindows = [],
  emphasis = false,
  metrics,
  onHover
}: MotionStripProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pinnedIdx, setPinnedIdx] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  
  const total = thumbnails.length;
  const totalFrames = total * frameStep;
  
  const handleMouseEnter = (idx: number, e: React.MouseEvent) => {
    setHoverIdx(idx);
    setMousePos({ x: e.clientX, y: e.clientY });
    const absoluteFrame = idx * frameStep;
    onHover?.(absoluteFrame);
  };
  
  const handleMouseLeave = () => {
    if (pinnedIdx === null) {
      setHoverIdx(null);
    }
  };
  
  const handleClick = (idx: number) => {
    setPinnedIdx(pinnedIdx === idx ? null : idx);
    setHoverIdx(idx);
  };
  
  const handleKeyDown = (e: React.KeyboardEvent, idx: number) => {
    if (e.key === 'ArrowLeft' && idx > 0) {
      setHoverIdx(idx - 1);
      const absoluteFrame = (idx - 1) * frameStep;
      onHover?.(absoluteFrame);
    } else if (e.key === 'ArrowRight' && idx < total - 1) {
      setHoverIdx(idx + 1);
      const absoluteFrame = (idx + 1) * frameStep;
      onHover?.(absoluteFrame);
    } else if (e.key === 'Enter') {
      handleClick(idx);
    }
  };
  
  // Get segment info for current hover
  const getSegmentInfo = (frameIdx: number) => {
    const absoluteFrame = frameIdx * frameStep;
    const segment = band.find(s => absoluteFrame >= s.fromFrame && absoluteFrame <= s.toFrame);
    return segment;
  };
  
  // Get metrics for current hover
  const getMetricsInfo = (frameIdx: number) => {
    if (!metrics) return null;
    
    const absoluteFrame = frameIdx * frameStep;
    const info: Record<string, { v: number; a: number }> = {};
    
    metrics.joints.forEach(joint => {
      const velocities = metrics.l2Velocity[joint === 'pelvis' ? 'Hips' : joint] || [];
      const accelerations = metrics.l2Acceleration[joint === 'pelvis' ? 'Hips' : joint] || [];
      
      if (absoluteFrame < velocities.length) {
        info[joint] = {
          v: velocities[absoluteFrame],
          a: accelerations[absoluteFrame]
        };
      }
    });
    
    return info;
  };
  
  const activeIdx = pinnedIdx !== null ? pinnedIdx : hoverIdx;
  const segment = activeIdx !== null ? getSegmentInfo(activeIdx) : null;
  const metricsInfo = activeIdx !== null ? getMetricsInfo(activeIdx) : null;
  
  return (
    <div className="w-full mb-6">
      {/* Header with label and tooltip */}
      <div className="mb-2 flex items-center gap-3">
        <h4 className={`text-sm font-semibold ${emphasis ? 'text-blue-400 text-base' : 'text-gray-300'}`}>
          {label}
        </h4>
        
        {/* Tooltip display */}
        {activeIdx !== null && (
          <div className="flex-1 flex items-center gap-4 text-xs text-gray-400 bg-gray-800 rounded-lg px-3 py-1.5">
            <span className="font-mono">
              Frame: #{activeIdx * frameStep} ({((activeIdx * frameStep) / fps).toFixed(2)}s)
            </span>
            
            {segment && (
              <span className="flex items-center gap-1">
                <span 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: segment.color }}
                />
                <span>{segment.label}</span>
              </span>
            )}
            
            {metricsInfo && (
              <span className="flex gap-3 ml-auto">
                {Object.entries(metricsInfo).slice(0, 3).map(([joint, vals]) => (
                  <span key={joint} className="font-mono">
                    {joint}: v={vals.v.toFixed(3)} a={vals.a.toFixed(3)}
                  </span>
                ))}
              </span>
            )}
            
            {pinnedIdx !== null && (
              <span className="text-yellow-400 text-xs">📌 Pinned</span>
            )}
          </div>
        )}
      </div>
      
      {/* Frame grid */}
      <div 
        className={`
          grid grid-flow-col auto-cols-max gap-1 p-3 rounded-xl bg-gray-900/50 
          shadow-lg overflow-x-auto
          ${emphasis ? 'ring-2 ring-blue-400/50 shadow-blue-500/20' : ''}
        `}
        style={{ 
          scrollbarWidth: 'thin',
          scrollbarColor: '#374151 #1f2937'
        }}
      >
        {thumbnails.map((src, idx) => {
          const isActive = activeIdx === idx;
          const absoluteFrame = idx * frameStep;
          const currentSegment = getSegmentInfo(idx);
          
          return (
            <button
              key={idx}
              className={`
                relative size-16 md:size-20 rounded-lg overflow-hidden 
                border-2 transition-all duration-200 flex-shrink-0
                ${isActive 
                  ? 'border-blue-400 shadow-lg shadow-blue-500/30 scale-105' 
                  : 'border-gray-700 hover:border-gray-500'
                }
                focus:outline-none focus:ring-2 focus:ring-blue-400
              `}
              onMouseEnter={(e) => handleMouseEnter(idx, e)}
              onMouseLeave={handleMouseLeave}
              onClick={() => handleClick(idx)}
              onKeyDown={(e) => handleKeyDown(e, idx)}
              onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}
              aria-label={`Frame ${absoluteFrame}${currentSegment ? `, ${currentSegment.label}` : ''}`}
              tabIndex={0}
            >
              {/* Thumbnail or placeholder */}
              {src ? (
                <img 
                  src={src} 
                  alt={`Frame ${absoluteFrame}`}
                  loading="lazy"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full bg-gray-800 flex items-center justify-center">
                  <span className="text-xs text-gray-500 font-mono">{absoluteFrame}</span>
                </div>
              )}
              
              {/* Frame number overlay */}
              <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs font-mono py-0.5 text-center">
                {absoluteFrame}
              </div>
            </button>
          );
        })}
      </div>
      
      {/* Under-strip colored band */}
      <UnderStripBand
        totalSamples={total}
        segments={band}
        frameStep={frameStep}
        totalFrames={totalFrames}
        highlightWindows={highlightWindows}
      />
      
      {/* Enhanced tooltip */}
      {activeIdx !== null && (
          <FrameTooltip
            frameIndex={activeIdx ?? 0}
            frameStep={frameStep}
            fps={fps}
            segment={segment}
            metrics={metrics}
            position={mousePos}
            visible={activeIdx !== null}
          />
      )}
    </div>
  );
}
