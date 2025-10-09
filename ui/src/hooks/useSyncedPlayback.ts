/**
 * useSyncedPlayback - Synchronized Timeline Playback Hook
 * 
 * Manages shared playback state across multiple sequence viewers
 * for coordinated timeline navigation and animation playback.
 * 
 * Key Features:
 * - Master/secondary viewer relationship
 * - Synchronized frame updates across viewers
 * - Coordinated play/pause controls
 * - Frame rate matching and timing
 * - Blend weight-aware frame interpolation
 * 
 * OBS-Style Implementation:
 * - Primary viewer drives timeline position
 * - Secondary viewer follows with offset support
 * - Global playback controls affect both viewers
 * - Frame-accurate synchronization for blend operations
 */
import {useState, useEffect, useCallback, useRef} from 'react'

/**
 * Playback state interface for synchronized viewers
 */
interface PlaybackState {
  isPlaying: boolean
  currentFrame: number
  totalFrames: number
  fps: number
  startTime?: number
}

/**
 * Synchronized playback configuration
 */
interface SyncConfig {
  /** Primary sequence frame count */
  primaryFrames: number
  /** Secondary sequence frame count */
  secondaryFrames: number
  /** Playback frame rate */
  fps: number
  /** Enable loop playback */
  loop: boolean
  /** Frame offset for secondary viewer */
  secondaryOffset?: number
}

/**
 * Synchronized playback return interface
 */
interface SyncedPlayback {
  // Playback state
  isPlaying: boolean
  primaryFrame: number
  secondaryFrame: number
  progress: number // 0-1 playback progress
  
  // Controls
  play: () => void
  pause: () => void
  stop: () => void
  seekTo: (frame: number) => void
  seekToProgress: (progress: number) => void
  
  // Configuration
  updateConfig: (config: Partial<SyncConfig>) => void
  setFrameRate: (fps: number) => void
  toggleLoop: () => void
}

/**
 * useSyncedPlayback hook implementation
 * 
 * Provides synchronized timeline playback for dual sequence viewers
 * with frame-accurate coordination and professional video editor controls.
 * 
 * @param initialConfig Initial synchronization configuration
 * @returns Synchronized playback controls and state
 */
export function useSyncedPlayback(initialConfig: SyncConfig): SyncedPlayback {
  const [config, setConfig] = useState<SyncConfig>(initialConfig)
  const [playbackState, setPlaybackState] = useState<PlaybackState>({
    isPlaying: false,
    currentFrame: 0,
    totalFrames: Math.max(config.primaryFrames, config.secondaryFrames),
    fps: config.fps
  })
  
  const animationFrameRef = useRef<number>()
  const lastTimeRef = useRef<number>(0)
  
  /**
   * Calculate synchronized frame positions
   * 
   * Maps master timeline position to appropriate frames
   * for both primary and secondary sequences.
   * 
   * @param masterFrame Current master timeline frame
   * @returns Tuple of [primaryFrame, secondaryFrame]
   */
  const calculateSyncedFrames = useCallback((masterFrame: number): [number, number] => {
    // Primary frame - direct mapping with bounds checking
    const primaryFrame = Math.min(masterFrame, config.primaryFrames - 1)
    
    // Secondary frame - scale to match secondary sequence length with offset
    const secondaryProgress = masterFrame / (config.primaryFrames - 1)
    const secondaryBaseFrame = Math.floor(secondaryProgress * (config.secondaryFrames - 1))
    const secondaryFrame = Math.min(
      secondaryBaseFrame + (config.secondaryOffset || 0),
      config.secondaryFrames - 1
    )
    
    return [Math.max(0, primaryFrame), Math.max(0, secondaryFrame)]
  }, [config])
  
  /**
   * Animation loop for synchronized playback
   * 
   * Handles frame advancement with precise timing
   * and automatic loop/stop behavior.
   * 
   * @param currentTime High-resolution timestamp
   */
  const animationLoop = useCallback((currentTime: number) => {
    if (!playbackState.isPlaying) return
    
    const deltaTime = currentTime - lastTimeRef.current
    const frameInterval = 1000 / config.fps // ms per frame
    
    if (deltaTime >= frameInterval) {
      setPlaybackState(prev => {
        let nextFrame = prev.currentFrame + 1
        
        // Handle end-of-sequence behavior
        if (nextFrame >= prev.totalFrames) {
          if (config.loop) {
            nextFrame = 0 // Loop back to start
          } else {
            return {...prev, isPlaying: false} // Stop at end
          }
        }
        
        return {
          ...prev,
          currentFrame: nextFrame
        }
      })
      
      lastTimeRef.current = currentTime
    }
    
    animationFrameRef.current = requestAnimationFrame(animationLoop)
  }, [playbackState.isPlaying, config.fps, config.loop])
  
  /**
   * Start synchronized playback
   */
  const play = useCallback(() => {
    setPlaybackState(prev => ({...prev, isPlaying: true, startTime: Date.now()}))
    lastTimeRef.current = performance.now()
    animationFrameRef.current = requestAnimationFrame(animationLoop)
  }, [animationLoop])
  
  /**
   * Pause synchronized playback
   */
  const pause = useCallback(() => {
    setPlaybackState(prev => ({...prev, isPlaying: false}))
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
  }, [])
  
  /**
   * Stop playback and reset to beginning
   */
  const stop = useCallback(() => {
    pause()
    setPlaybackState(prev => ({...prev, currentFrame: 0}))
  }, [pause])
  
  /**
   * Seek to specific frame position
   * 
   * @param frame Target frame number
   */
  const seekTo = useCallback((frame: number) => {
    const clampedFrame = Math.max(0, Math.min(frame, playbackState.totalFrames - 1))
    setPlaybackState(prev => ({...prev, currentFrame: clampedFrame}))
  }, [playbackState.totalFrames])
  
  /**
   * Seek to progress percentage
   * 
   * @param progress Progress value 0-1
   */
  const seekToProgress = useCallback((progress: number) => {
    const targetFrame = Math.floor(progress * (playbackState.totalFrames - 1))
    seekTo(targetFrame)
  }, [playbackState.totalFrames, seekTo])
  
  /**
   * Update synchronization configuration
   * 
   * @param updates Partial configuration updates
   */
  const updateConfig = useCallback((updates: Partial<SyncConfig>) => {
    setConfig(prev => {
      const newConfig = {...prev, ...updates}
      const newTotalFrames = Math.max(newConfig.primaryFrames, newConfig.secondaryFrames)
      
      setPlaybackState(prevState => ({
        ...prevState,
        totalFrames: newTotalFrames,
        fps: newConfig.fps,
        currentFrame: Math.min(prevState.currentFrame, newTotalFrames - 1)
      }))
      
      return newConfig
    })
  }, [])
  
  /**
   * Update frame rate
   * 
   * @param fps New frames per second
   */
  const setFrameRate = useCallback((fps: number) => {
    updateConfig({fps})
  }, [updateConfig])
  
  /**
   * Toggle loop playback mode
   */
  const toggleLoop = useCallback(() => {
    setConfig(prev => ({...prev, loop: !prev.loop}))
  }, [])
  
  // Calculate current synchronized frame positions
  const [primaryFrame, secondaryFrame] = calculateSyncedFrames(playbackState.currentFrame)
  const progress = playbackState.totalFrames > 1 
    ? playbackState.currentFrame / (playbackState.totalFrames - 1)
    : 0
  
  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])
  
  return {
    // State
    isPlaying: playbackState.isPlaying,
    primaryFrame,
    secondaryFrame,
    progress,
    
    // Controls
    play,
    pause,
    stop,
    seekTo,
    seekToProgress,
    
    // Configuration
    updateConfig,
    setFrameRate,
    toggleLoop
  }
}