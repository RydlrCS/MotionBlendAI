import React from 'react'

export default function MotionList({motions, selected, onToggle} : any){
  return (
    <div className="motion-list">
      <h3>Available motions</h3>
      <ul>
        {motions.map((m: any) => (
          <li key={m.id} className={selected.includes(m.id) ? 'selected' : ''}>
            <label>
              <input type="checkbox" checked={selected.includes(m.id)} onChange={()=>onToggle(m.id)} />
              <span className="name">{m.name}</span>
            </label>
          </li>
        ))}
      </ul>
    </div>
  )
}
