import React from 'react'

type Props = { values: number[]; width?: number; height?: number }

export default function Sparkline({values, width=200, height=40}: Props){
  if(!values || values.length === 0) return <svg width={width} height={height}><text x={4} y={height/2}>no data</text></svg>
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const step = width / (values.length - 1 || 1)
  const points = values.map((v,i)=>`${i*step},${height - ((v-min)/range)*height}`).join(' ')
  return (
    <svg width={width} height={height}>
      <polyline points={points} fill="none" stroke="#2563eb" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}
