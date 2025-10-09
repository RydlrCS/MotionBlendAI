const express = require('express')
const app = express()
app.use(express.json())

// Mock motion data with frame information for visualization
const motions = [
  {
    id: 'm001',
    name: 'walk_fwd',
    formats: ['npy', 'fbx'],
    duration: 2.5,
    shape: [75, 16, 3], // 75 frames, 16 joints, 3D coordinates
    frames: generateMockFrames(75, 16), // Generate sample frame data
    jointNames: [
      'root', 'spine1', 'spine2', 'head',
      'left_shoulder', 'left_elbow', 'left_wrist',
      'right_shoulder', 'right_elbow', 'right_wrist',
      'left_hip', 'left_knee', 'left_ankle',
      'right_hip', 'right_knee', 'right_ankle'
    ]
  },
  {
    id: 'm002',
    name: 'run_right',
    formats: ['npy', 'fbx'],
    duration: 1.6,
    shape: [48, 16, 3], // 48 frames, 16 joints, 3D coordinates
    frames: generateMockFrames(48, 16),
    jointNames: [
      'root', 'spine1', 'spine2', 'head',
      'left_shoulder', 'left_elbow', 'left_wrist',
      'right_shoulder', 'right_elbow', 'right_wrist',
      'left_hip', 'left_knee', 'left_ankle',
      'right_hip', 'right_knee', 'right_ankle'
    ]
  },
  {
    id: 'm003',
    name: 'jump_up',
    formats: ['npy', 'fbx'],
    duration: 0.9,
    shape: [27, 16, 3], // 27 frames, 16 joints, 3D coordinates
    frames: generateMockFrames(27, 16),
    jointNames: [
      'root', 'spine1', 'spine2', 'head',
      'left_shoulder', 'left_elbow', 'left_wrist',
      'right_shoulder', 'right_elbow', 'right_wrist',
      'left_hip', 'left_knee', 'left_ankle',
      'right_hip', 'right_knee', 'right_ankle'
    ]
  }
]

/**
 * Generate mock motion capture frame data
 * Creates realistic-looking joint positions for demonstration
 * 
 * @param {number} frameCount - Number of frames to generate
 * @param {number} jointCount - Number of joints per frame
 * @returns {Array} Array of frames with joint positions
 */
function generateMockFrames(frameCount, jointCount) {
  const frames = []
  
  for (let frame = 0; frame < frameCount; frame++) {
    const joints = []
    const time = frame / frameCount * Math.PI * 2 // Animation cycle
    
    for (let joint = 0; joint < jointCount; joint++) {
      // Create realistic motion patterns
      const baseY = joint < 4 ? 1.0 : 0.5 // Upper body higher than lower
      const walkCycle = Math.sin(time * 2 + joint) * 0.1
      const bounce = Math.sin(time * 4) * 0.05
      
      joints.push({
        x: Math.cos(time + joint * 0.1) * 0.3 + Math.sin(time * 3) * 0.1,
        y: baseY + walkCycle + bounce,
        z: Math.sin(time * 1.5 + joint * 0.2) * 0.2,
        name: `joint_${joint}`
      })
    }
    
    frames.push(joints)
  }
  
  return frames
}

// GET /api/motions - Return available motion sequences with frame data
app.get('/api/motions', (req, res) => {
  const motions = [
    {
      id: 'm001',
      name: 'walk_fwd', 
      formats: ['npy', 'fbx'],
      duration: 2.5,
      shape: [75, 22, 3], // 75 frames, 22 joints, 3D coordinates
      joints: 22,
      frames: 75,
      fps: 30
    },
    {
      id: 'm002',
      name: 'run_right',
      formats: ['npy', 'fbx'], 
      duration: 1.6,
      shape: [48, 22, 3],
      joints: 22,
      frames: 48, 
      fps: 30
    },
    {
      id: 'm003',
      name: 'jump_up',
      formats: ['npy', 'fbx'],
      duration: 0.9,
      shape: [27, 22, 3],
      joints: 22,
      frames: 27,
      fps: 30
    }
  ]
  res.json({motions})
})

app.post('/api/blend', (req,res)=>{
  const id = 'job-'+Date.now()
  res.status(202).json({job_id: id, status:'QUEUED', submitted_at: new Date().toISOString()})
})

app.get('/api/jobs/:id', (req,res)=>{
  res.json({job_id:req.params.id, status:'SUCCEEDED', result:{artifacts:[{path:'gs://bucket/out.npy', name:'blend.npy'}]}})
})

// Serve demo artifacts (files produced by scripts/run_demo.sh)
const path = require('path')
const artifactsDir = path.join(__dirname, '..', 'build', 'demo_artifacts')
app.use('/artifacts', express.static(artifactsDir))
app.get('/api/artifacts', (req, res)=>{
  const fs = require('fs')
  try{
    const files = fs.existsSync(artifactsDir) ? fs.readdirSync(artifactsDir).filter(f=>!f.startsWith('.')) : []
    const stat = (n) => {
      try{ const s = fs.statSync(path.join(artifactsDir, n)); return {size: s.size} }catch(e){return {size:0}}
    }
    res.json({artifacts: files.map(n=>({name:n, size: stat(n).size}))})
  }catch(e){
    res.status(500).json({error: String(e)})
  }
})

app.get('/api/artifact/:name/describe', (req, res)=>{
  const child = require('child_process')
  const path = require('path')
  const name = req.params.name
  const file = path.join(artifactsDir, name)
  if(!require('fs').existsSync(file)){
    return res.status(404).json({error:'not found'})
  }
  child.execFile('python3', [path.join(__dirname,'..','scripts','describe_npy.py'), file], {timeout:5000}, (err, stdout, stderr)=>{
    if(err){
      return res.status(500).json({error: String(err), stderr: stderr.toString()})
    }
    try{
  const obj = JSON.parse(stdout)
  // include file size
  try{ obj.size = require('fs').statSync(file).size }catch(e){}
  res.json(obj)
    }catch(e){
      res.status(500).json({error: 'invalid json', raw: stdout, stderr})
    }
  })
})

app.get('/api/artifacts/manifest', (req,res)=>{
  const fs = require('fs')
  const m = path.join(artifactsDir, 'manifest.json')
  if(!fs.existsSync(m)) return res.status(404).json({error:'no manifest'})
  try{ res.json(JSON.parse(fs.readFileSync(m,'utf8'))) }catch(e){ res.status(500).json({error:String(e)}) }
})

const port = process.env.PORT || 3000
app.listen(port, ()=> console.log('Mock server listening on', port))
