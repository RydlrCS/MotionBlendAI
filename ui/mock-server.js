const express = require('express')
const app = express()
app.use(express.json())

const motions = [
  {id:'m001', name:'walk_fwd', formats:['npy','fbx'], duration:2.5},
  {id:'m002', name:'run_right', formats:['npy','fbx'], duration:1.6},
  {id:'m003', name:'jump_up', formats:['npy','fbx'], duration:0.9}
]

app.get('/api/motions', (req,res)=>{
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
