/**
 * Comprehensive type definitions for motion blend artifacts with frame-grid strips,
 * colored timelines, and metrics visualization.
 */

export type JointName = "pelvis" | "lwrist" | "rwrist" | "lfoot" | "rfoot";

export type Segment = {
  fromFrame: number;
  toFrame: number;
  label: string;
  color: string;
  alpha?: number;
}

export type SourceMotion = {
  id: string;
  label: string;
  character?: string;
  frames: number;
  sampleEvery: number;
  thumbnails: string[];
  color: string;
}

export type BlendMotion = {
  id: string;
  label: string;
  frames: number;
  sampleEvery: number;
  thumbnails: string[];
  segments: Segment[];
}

export type ArtifactMetrics = {
  joints: JointName[];
  l2Velocity: Record<string, number[]>;
  l2Acceleration: Record<string, number[]>;
  fid?: number;
  cov?: number;
  gdiv?: number;
  ldiv?: number;
  interDiv?: number;
  intraDiv?: number;
  transitionWindows?: { start: number; end: number; }[];
}

export type Artifact = {
  id: string;
  name: string;
  createdAt: string;
  fps: number;
  frames: number;
  sources?: SourceMotion[];
  blend?: BlendMotion;
  metrics: ArtifactMetrics;
  files?: {
    previewPng?: string;
    previewMp4?: string;
    metricsJson?: string;
  };
  type?: string;
  status?: string;
  metadata?: any;
  analysis?: any;
  description?: string;
  file_path?: string;
}

export type MotionStripProps = {
  label: string;
  thumbnails: string[];
  band: Segment[];
  fps: number;
  frameStep: number;
  onHover?: (frameIdx: number) => void;
  highlightWindows?: { start:number; end:number; }[];
  emphasis?: boolean;
  metrics?: ArtifactMetrics;
  showMetrics?: boolean;
  selectedJoints?: string[];
}

export type UnderStripBandProps = {
  totalSamples: number;
  segments: Segment[];
  frameStep: number;
  totalFrames: number;
  highlightWindows?: { start:number; end:number; }[];
}

export type FrameGridProps = {
  thumbnails: string[];
  frameStep: number;
  fps: number;
  onHover?: (index: number, frameNumber: number) => void;
  hoveredIndex: number | null;
  emphasis?: boolean;
  metrics?: ArtifactMetrics;
  selectedJoints?: string[];
  currentFrame?: number;
}

export type FrameTooltipData = {
  frameIndex: number;
  timeSeconds: number;
  segmentLabel: string;
  metrics?: {
    [joint: string]: {
      velocity: number;
      acceleration: number;
    }
  }
}

export type ExportOptions = {
  png?: boolean;
  csv?: boolean;
  json?: boolean;
}

export const MOTION_COLORS = {
  salsa: '#2FBF71',
  swing: '#F0A202',
  wave: '#2D7DD2',
  capoeira: '#41B883',
  break: '#FF6B6B',
  walking: '#14B8A6',
  running: '#F59E0B',
  jumping: '#8B5CF6',
  dancing: '#EC4899',
  fighting: '#EF4444',
  default: '#6B7280'
} as const;

export const DEFAULT_COLORS = [
  '#2FBF71', '#F0A202', '#2D7DD2', '#41B883', '#FF6B6B',
  '#9333EA', '#EC4899', '#14B8A6', '#F59E0B', '#8B5CF6'
];

export function getMotionColor(index: number): string {
  return DEFAULT_COLORS[index % DEFAULT_COLORS.length];
}

export const JOINT_LABELS: Record<string, string> = {
  pelvis: 'Pelvis',
  lwrist: 'Left Wrist',
  rwrist: 'Right Wrist',
  lfoot: 'Left Foot',
  rfoot: 'Right Foot',
  Hips: 'Pelvis',
  LeftWrist: 'Left Wrist',
  RightWrist: 'Right Wrist',
  LeftFoot: 'Left Foot',
  RightFoot: 'Right Foot',
};

export const SAMPLE_RATES = {
  fine: 6,
  medium: 10,
  coarse: 15,
  veryCoarse: 20,
} as const;
