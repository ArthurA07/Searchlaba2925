import React, { useEffect, useMemo, useRef, useState } from 'react'

type Detection = { name: string; score: number; present?: boolean; bbox?: number[] }
type RLE = { counts: number[]; size: [number, number] }
type InferV1 = { detections: (Detection & { mask_rle?: RLE })[]; image_size: [number, number]; used_imgsz: number; roi_bbox?: number[]; infer_ms: number; saved_json?: string; saved_viz?: string }
type MatchV1 = { results: { name: string; present: boolean; score: number }[]; manual_recount: boolean; threshold: number }

const API_V1 = (import.meta.env.VITE_API_URL_V1 as string) ?? 'http://localhost:8000/api/v1'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [threshold, setThreshold] = useState(0.9)
  const [infer, setInfer] = useState<InferV1 | null>(null)
  const [match, setMatch] = useState<MatchV1 | null>(null)
  const [loading, setLoading] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const drawRLE = (ctx: CanvasRenderingContext2D, rle: RLE, scale: number) => {
    const [h, w] = rle.size
    const imgData = ctx.createImageData(Math.floor(w * scale), Math.floor(h * scale))
    // Decode RLE (uncompressed counts starting with background)
    let val = 0
    let idx = 0
    for (const run of rle.counts) {
      for (let k = 0; k < run; k++) {
        const y = Math.floor(idx / w)
        const x = idx % w
        if (val === 1) {
          // draw scaled pixel as white with alpha
          for (let dy = 0; dy < scale; dy++) {
            for (let dx = 0; dx < scale; dx++) {
              const sx = Math.floor(x * scale) + dx
              const sy = Math.floor(y * scale) + dy
              const p = (sy * Math.floor(w * scale) + sx) * 4
              imgData.data[p] = 255; imgData.data[p + 1] = 255; imgData.data[p + 2] = 255; imgData.data[p + 3] = 90
            }
          }
        }
        idx++
        if (idx >= w * h) break
      }
      val = 1 - val
      if (idx >= w * h) break
    }
    ctx.putImageData(imgData, 0, 0)
  }

  useEffect(() => {
    if (!infer || infer.detections.length === 0) return
    const cnv = canvasRef.current
    if (!cnv) return
    const [W, H] = infer.image_size
    const maxW = 640
    const scale = Math.max(1, Math.floor(Math.min(maxW / W, maxW / H) * 1))
    cnv.width = Math.floor(W * scale)
    cnv.height = Math.floor(H * scale)
    const ctx = cnv.getContext('2d')!
    ctx.clearRect(0, 0, cnv.width, cnv.height)
    infer.detections.forEach(d => {
      if (d.mask_rle) drawRLE(ctx, d.mask_rle, scale)
    })
  }, [infer])

  const formData = useMemo(() => {
    const fd = new FormData()
    if (file) fd.append('file', file)
    return fd
  }, [file])

  const onInfer = async () => {
    if (!file) return
    setLoading(true)
    setInfer(null)
    setMatch(null)
    try {
      const inferRes = await fetch(`${API_V1}/infer`, { method: 'POST', body: formData })
      const inferJson = (await inferRes.json()) as InferV1
      setInfer(inferJson)
      const matchRes = await fetch(`${API_V1}/match?threshold=${threshold}`, { method: 'POST', body: formData })
      const matchJson = (await matchRes.json()) as MatchV1
      setMatch(matchJson)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 960, margin: '24px auto', fontFamily: 'Inter, system-ui, Arial' }}>
      <h2>CV MVP — Выдача/Приёмка</h2>
      <div style={{ display: 'grid', gap: 12 }}>
        <input type="file" accept="image/*" onChange={e => setFile(e.target.files?.[0] ?? null)} />
        <label>
          Порог: {threshold.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} />
        </label>
        <button onClick={onInfer} disabled={!file || loading}>{loading ? 'Обработка…' : 'Определить'}</button>
      </div>

      {infer && (
        <div style={{ marginTop: 16 }}>
          <h3>Предпросмотр ROI и детекции</h3>
          <div>Исходный размер: {infer.image_size.join('×')} | used_imgsz: {infer.used_imgsz} | инференс: {infer.infer_ms.toFixed(1)} ms</div>
          {infer.roi_bbox && (
            <div>ROI bbox: [{infer.roi_bbox.join(', ')}]</div>
          )}
          <div style={{marginTop:8}}>
            <canvas ref={canvasRef} style={{maxWidth:'100%', border:'1px solid #eee'}} />
          </div>
        </div>
      )}

      {match && (
        <div style={{ marginTop: 16 }}>
          <h3>Совпадение по 11 классам</h3>
          <div>
            Порог: {match.threshold} | Ручной пересчёт: {match.manual_recount ? 'Да' : 'Нет'} |
            Match %: {Math.round(100 * (match.results.filter(r => r.present).length / Math.max(1, match.results.length)))}%
          </div>
          <table style={{ borderCollapse: 'collapse', marginTop: 12 }}>
            <thead>
              <tr>
                <th style={{ border: '1px solid #ccc', padding: 6 }}>Класс</th>
                <th style={{ border: '1px solid #ccc', padding: 6 }}>present</th>
                <th style={{ border: '1px solid #ccc', padding: 6 }}>score</th>
              </tr>
            </thead>
            <tbody>
              {match.results.map(t => (
                <tr key={t.id}>
                  <td style={{ border: '1px solid #ccc', padding: 6 }}>{t.name}</td>
                  <td style={{ border: '1px solid #ccc', padding: 6, color: t.present ? '#0a0' : '#a00' }}>{t.present ? 'true' : 'false'}</td>
                  <td style={{ border: '1px solid #ccc', padding: 6 }}>{t.score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {infer?.saved_json && (
            <div style={{ marginTop: 8 }}>
              <a href={infer.saved_json} download>Скачать JSON</a>
            </div>
          )}
        </div>
      )}
    </div>
  )
}


