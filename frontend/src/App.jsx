import React, { useState, useEffect, useRef } from 'react'
import {
  Satellite,
  Layers,
  Search,
  Upload,
  FileCheck,
  MapPin,
  FileText,
  Activity,
  Download,
  Clock,
  Sparkles,
  Info,
  CheckCircle2,
  Thermometer,
  Droplets,
  Mountain,
  Compass,
  TreePine,
  Building2,
  Eye,
  Moon,
  Sun
} from 'lucide-react'
import L from 'leaflet'

export default function App() {
  const [mode, setMode] = useState('single')
  const [files, setFiles] = useState([])
  const [query, setQuery] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [result, setResult] = useState(null)
  const [activeTab, setActiveTab] = useState('evidence')
  const [selectedEvidenceIdx, setSelectedEvidenceIdx] = useState(0)
  const [serverHealth, setServerHealth] = useState(null)
  const [sessionAnalysisId, setSessionAnalysisId] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('satquery-theme') || 'light')
  
  const mapRef = useRef(null)
  const leafletInstance = useRef(null)

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => setServerHealth(d))
      .catch(() => setServerHealth({ status: 'offline' }))
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('satquery-theme', theme)
  }, [theme])

  useEffect(() => {
    let interval = null
    if (isAnalyzing) {
      setElapsed(0)
      interval = setInterval(() => setElapsed(e => e + 1), 1000)
    } else {
      clearInterval(interval)
    }
    return () => clearInterval(interval)
  }, [isAnalyzing])

  useEffect(() => {
    if (activeTab === 'map' && result?.map?.georeferenced) {
      const bounds = result.map.bounds
      const center = result.map.center
      if (!bounds || !center) return

      setTimeout(() => {
        if (!mapRef.current) return
        if (leafletInstance.current) {
          leafletInstance.current.remove()
        }

        const map = L.map(mapRef.current).setView([center[0], center[1]], 13)
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map)

        const latLngBounds = [
          [bounds.min_lat, bounds.min_lon],
          [bounds.max_lat, bounds.max_lon]
        ]
        L.rectangle(latLngBounds, { color: '#38bdf8', weight: 2, fillOpacity: 0.25 }).addTo(map)
        
        const locName = result.location?.country ? `${result.location.nearest_city || result.location.state}, ${result.location.country}` : 'Satellite Footprint'
        L.marker([center[0], center[1]]).addTo(map)
          .bindPopup(`<b>${locName}</b><br>Lat: ${center[0].toFixed(4)}°, Lon: ${center[1].toFixed(4)}°<br>CRS: ${result.map.crs}`)
          .openPopup()

        map.fitBounds(latLngBounds)
        leafletInstance.current = map
      }, 150)
    }
  }, [activeTab, result])

  const loadSampleDataset = async (type) => {
    setFiles([])
    setResult(null)
    setIsAnalyzing(true)

    try {
      if (type === 'water_grounding') {
        setMode('single')
        setQuery('Give me information about this image and highlight the water body.')
      } else if (type === 'optical_sar') {
        setMode('optical_sar')
        setQuery('Compare optical and SAR images to identify water and built-up structures')
      } else if (type === 'bitemporal_change') {
        setMode('bitemporal')
        setQuery('What changed between these images? Has the built-up area increased?')
      } else if (type === 'non_geo') {
        setMode('single')
        setQuery('Give me information about this image.')
      }

      const res = await fetch('/api/sample-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_type: type })
      })
      const data = await res.json()
      setSessionAnalysisId(data.analysis_id)
      setFiles(data.files.map(f => ({
        name: f.filename,
        size: `${(f.dimensions[0] * f.dimensions[1] * f.bands / (1024 * 1024)).toFixed(2)} MB`,
        format: f.format,
        crs: f.crs
      })))
    } catch (err) {
      alert(`Could not load sample: ${err.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files)
    if (!selected.length) return
    setSessionAnalysisId(null)
    setFiles(selected.map(f => ({
      name: f.name,
      size: `${(f.size / (1024 * 1024)).toFixed(2)} MB`,
      fileObject: f
    })))
  }

  const handleAnalyze = async () => {
    if (!query.trim()) return
    setIsAnalyzing(true)
    setResult(null)

    try {
      let analysisId = sessionAnalysisId
      let filenames = []

      const hasRawFiles = files.some(f => f.fileObject)
      if (hasRawFiles) {
        const formData = new FormData()
        files.forEach(f => {
          if (f.fileObject) formData.append('files', f.fileObject)
        })
        const upRes = await fetch('/api/upload', { method: 'POST', body: formData })
        if (!upRes.ok) {
          const err = await upRes.json()
          throw new Error(err.detail || 'Upload failed')
        }
        const upData = await upRes.json()
        analysisId = upData.analysis_id
        filenames = upData.files.map(f => f.filename)
      } else {
        filenames = files.map(f => f.name)
      }

      if (!analysisId) {
        throw new Error('Please upload an image or select a verified SIH dataset.')
      }

      const analyzeForm = new FormData()
      analyzeForm.append('analysis_id', analysisId)
      analyzeForm.append('query', query)
      analyzeForm.append('filenames', filenames.join(','))

      const aRes = await fetch('/api/analyze', { method: 'POST', body: analyzeForm })
      if (!aRes.ok) {
        const err = await aRes.json()
        throw new Error(err.detail || 'Analysis failed')
      }
      const aData = await aRes.json()
      setResult(aData)
      setActiveTab('intelligence')
      setSelectedEvidenceIdx(0)
    } catch (err) {
      alert(`Analysis notice: ${err.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div>
          <h1 className="brand-title">
            <Satellite size={28} /> SATQUERY AI
          </h1>
          <p className="brand-subtitle">
            Multimodal Remote Sensing & Environmental Intelligence | Smart India Hackathon 2026 (SIH26167)
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            className="theme-toggle"
            onClick={() => setTheme(currentTheme => currentTheme === 'light' ? 'dark' : 'light')}
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
            <span>{theme === 'light' ? 'Dark' : 'Light'}</span>
          </button>
          <span className="badge badge-cyan">ISRO / SAC Remote Sensing Engine</span>
          <span className={`badge ${serverHealth?.status === 'healthy' ? 'badge-emerald' : 'badge-amber'}`}>
            <Activity size={12} /> {serverHealth?.status === 'healthy' ? 'Pure Ephemeral Runtime (No-DB)' : 'Connecting...'}
          </span>
        </div>
      </header>

      {/* Mode Selector */}
      <div className="mode-selector">
        <button
          className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
          onClick={() => { setMode('single'); setFiles([]); setResult(null); setSessionAnalysisId(null); }}
        >
          <Satellite size={16} /> Single Satellite Image (VQA / Grounding / Intelligence)
        </button>
        <button
          className={`mode-btn ${mode === 'optical_sar' ? 'active' : ''}`}
          onClick={() => { setMode('optical_sar'); setFiles([]); setResult(null); setSessionAnalysisId(null); }}
        >
          <Layers size={16} /> Optical + SAR Pair (Cross-Modal Fusion)
        </button>
        <button
          className={`mode-btn ${mode === 'bitemporal' ? 'active' : ''}`}
          onClick={() => { setMode('bitemporal'); setFiles([]); setResult(null); setSessionAnalysisId(null); }}
        >
          <Clock size={16} /> Bi-Temporal Pair (Change Detection)
        </button>
      </div>

      {/* Upload Zone Card */}
      <div className="card">
        <div className="card-title">
          <Upload size={16} /> Remote Sensing Imagery Input
        </div>

        <label className="dropzone" style={{ display: 'block' }}>
          <input
            type="file"
            multiple={mode !== 'single'}
            accept=".tif,.tiff,.png,.jpg,.jpeg"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <Satellite size={34} color="var(--accent-cyan)" style={{ margin: '0 auto 10px' }} />
          <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
            {mode === 'single' && 'Upload Single Satellite Image (GeoTIFF / PNG / JPEG)'}
            {mode === 'optical_sar' && 'Upload Optical + SAR Pair (2 Images)'}
            {mode === 'bitemporal' && 'Upload Bi-Temporal Images (T1 Before, T2 After)'}
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '6px' }}>
            Supports multi-band GeoTIFF with full CRS, affine transform, sensor metadata, and NoData preservation.
          </p>
        </label>

        {files.length > 0 && (
          <div style={{ marginTop: '16px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {files.map((f, i) => (
              <div key={i} style={{
                background: 'var(--surface-soft)',
                border: '1px solid var(--bg-card-border)',
                padding: '8px 14px',
                borderRadius: '6px',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <FileCheck size={14} color="#10b981" />
                <strong>{f.name}</strong>
                <span style={{ color: 'var(--text-muted)' }}>({f.size})</span>
                {f.crs && <span className="badge badge-cyan" style={{ fontSize: '10px' }}>{f.crs}</span>}
              </div>
            ))}
          </div>
        )}

        {/* Quick Sample Presets */}
        <div className="sample-bar">
          <span style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Sparkles size={12} /> Try Verified SIH Datasets:
          </span>
          <button className="sample-chip" onClick={() => loadSampleDataset('water_grounding')}>
            🌊 Delhi UTM 4-band GeoTIFF (Water + NDVI)
          </button>
          <button className="sample-chip" onClick={() => loadSampleDataset('optical_sar')}>
            🛰️ Optical + RISAT SAR Pair
          </button>
          <button className="sample-chip" onClick={() => loadSampleDataset('bitemporal_change')}>
            ⏳ Bi-Temporal Urban Change
          </button>
          <button className="sample-chip" onClick={() => loadSampleDataset('non_geo')}>
            🖼️ Non-Georeferenced PNG (Zero-Fake-Coords Test)
          </button>
        </div>

        {/* Natural Language Query Bar */}
        <div className="query-box">
          <input
            type="text"
            className="query-input"
            placeholder={
              mode === 'single' ? 'Ask a question e.g. "Give me information about this image" or "Where is this image?"' :
              mode === 'optical_sar' ? 'e.g. "Compare optical and SAR images to identify water and built-up structures"' :
              'e.g. "What changed between these two images?" or "Has the built-up area increased?"'
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
          />
          <button
            className="btn-primary"
            onClick={handleAnalyze}
            disabled={isAnalyzing || files.length === 0 || !query.trim()}
          >
            {isAnalyzing ? (
              <>
                <Activity className="animate-spin" size={16} /> Analyzing ({elapsed}s)...
              </>
            ) : (
              <>
                <Search size={16} /> ANALYZE
              </>
            )}
          </button>
        </div>
      </div>

      {/* Analysis Output Section */}
      {result && (
        <div>
          {/* Executive Analysis Card */}
          <div className="card">
            <div className="card-title">
              <Sparkles size={16} /> Executive Remote Sensing Briefing
            </div>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '14px', alignItems: 'center', flexWrap: 'wrap' }}>
              <span className="badge badge-cyan">Task: {result.task}</span>
              {result.confidence !== null ? (
                <span className="badge badge-emerald">
                  Model Confidence: {(result.confidence * 100).toFixed(1)}% (Verified)
                </span>
              ) : (
                <span className="badge badge-amber">Algorithm-derived Metric</span>
              )}
              {result.report_url && (
                <a
                  href={result.report_url}
                  target="_blank"
                  rel="noreferrer"
                  className="badge badge-cyan"
                  style={{ textDecoration: 'none', marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '4px' }}
                >
                  <Download size={12} /> Download Standalone Report
                </a>
              )}
            </div>

            <div className="answer-card">
              <p style={{ color: 'var(--text-primary)', fontWeight: 500, fontSize: '15px' }}>{result.answer}</p>
            </div>

            {/* Navigation Tabs */}
            <div className="tabs">
              <button
                className={`tab-btn ${activeTab === 'intelligence' ? 'active' : ''}`}
                onClick={() => setActiveTab('intelligence')}
              >
                <Compass size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Full Scene Intelligence
              </button>
              <button
                className={`tab-btn ${activeTab === 'evidence' ? 'active' : ''}`}
                onClick={() => setActiveTab('evidence')}
              >
                <Layers size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Visual Evidence Gallery ({result.visual_evidence?.length || 0})
              </button>
              <button
                className={`tab-btn ${activeTab === 'map' ? 'active' : ''}`}
                onClick={() => setActiveTab('map')}
              >
                <MapPin size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Geospatial Map View
              </button>
              <button
                className={`tab-btn ${activeTab === 'trace' ? 'active' : ''}`}
                onClick={() => setActiveTab('trace')}
              >
                <FileText size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Auditable Execution & Provenance
              </button>
            </div>

            {/* Tab 1: Full Scene Intelligence */}
            {activeTab === 'intelligence' && (
              <div>
                {/* 1. Location & Image Metadata Grid */}
                <div className="intelligence-grid">
                  <div className="stat-item">
                    <div className="stat-label"><MapPin size={12} style={{ verticalAlign: 'middle' }} /> Geographic Location</div>
                    {result.location?.status === 'available' ? (
                      <div>
                        <div className="stat-value">{result.location.nearest_city || result.location.state}, {result.location.country}</div>
                        <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                          Lat: {result.location.latitude}° N, Lon: {result.location.longitude}° E ({result.location.continent})
                        </div>
                        <div className="stat-source">Source: {result.location.provenance?.source}</div>
                      </div>
                    ) : (
                      <div>
                        <div className="stat-value" style={{ color: '#94a3b8' }}>Unavailable</div>
                        <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                          {result.location?.reason || 'Non-georeferenced imagery'}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="stat-item">
                    <div className="stat-label"><Satellite size={12} style={{ verticalAlign: 'middle' }} /> Sensor & Acquisition</div>
                    <div className="stat-value">{result.metadata?.sensor || 'Remote Sensing Satellite'}</div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                      Acquisition: {result.metadata?.acquisition_date || 'Standard capture'} | Res: {result.metadata?.resolution_meters}m/px
                    </div>
                    <div className="stat-source">CRS: {result.metadata?.crs || 'Non-georeferenced'} ({result.metadata?.bands} bands)</div>
                  </div>

                  <div className="stat-item">
                    <div className="stat-label"><Thermometer size={12} style={{ verticalAlign: 'middle' }} /> Surface Temperature</div>
                    {result.environment?.temperature?.status === 'available' ? (
                      <div>
                        <div className="stat-value">{result.environment.temperature.value} {result.environment.temperature.unit}</div>
                        <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                          Acquisition Sample: {result.environment.temperature.sample_time_utc}
                        </div>
                        <div className="stat-source">Source: {result.environment.temperature.source}</div>
                      </div>
                    ) : (
                      <div>
                        <div className="stat-value" style={{ color: '#94a3b8' }}>Unavailable</div>
                        <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                          {result.environment?.temperature?.reason || 'Requires georeferenced coordinates'}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="stat-item">
                    <div className="stat-label"><Droplets size={12} style={{ verticalAlign: 'middle' }} /> Surface Soil Moisture</div>
                    {result.environment?.soil_moisture?.status === 'available' ? (
                      <div>
                        <div className="stat-value">{result.environment.soil_moisture.value} {result.environment.soil_moisture.unit}</div>
                        <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                          Depth: {result.environment.soil_moisture.depth} ({result.environment.soil_moisture.spatial_resolution})
                        </div>
                        <div className="stat-source">Source: {result.environment.soil_moisture.source}</div>
                      </div>
                    ) : (
                      <div>
                        <div className="stat-value" style={{ color: '#94a3b8' }}>Unavailable</div>
                        <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                          {result.environment?.soil_moisture?.reason || 'Requires georeferenced coordinates'}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="stat-item">
                    <div className="stat-label"><Mountain size={12} style={{ verticalAlign: 'middle' }} /> Topography & Elevation</div>
                    <div className="stat-value">{result.terrain?.elevation_meters !== null ? `${result.terrain.elevation_meters} m` : 'Unavailable'}</div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                      {result.terrain?.topography || 'Plains / River Basin'}
                    </div>
                    <div className="stat-source">Source: {result.terrain?.provenance?.source || 'Digital Elevation Model'}</div>
                  </div>

                  <div className="stat-item">
                    <div className="stat-label"><TreePine size={12} style={{ verticalAlign: 'middle' }} /> Vegetation & NDVI</div>
                    <div className="stat-value">
                      {result.vegetation?.ndvi_available ? `Mean NDVI: ${result.vegetation.mean_ndvi}` : `${result.vegetation?.coverage_pct}% (Visible ExG)`}
                    </div>
                    <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                      {result.vegetation?.condition}
                    </div>
                    <div className="stat-source">Method: {result.vegetation?.provenance?.method}</div>
                  </div>
                </div>

                {/* 2. Land Cover Breakdown Progress Bars */}
                <div style={{ background: '#090e1a', border: '1px solid #1e293b', borderRadius: '8px', padding: '18px 20px', marginTop: '16px' }}>
                  <h4 style={{ fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', marginBottom: '14px' }}>
                    Quantitative Multi-Class Land Cover Breakdown
                  </h4>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                        <span style={{ color: '#22c55e' }}>🌲 Total Vegetation</span>
                        <strong>{result.land_cover?.classes?.total_vegetation_pct}%</strong>
                      </div>
                      <div className="bar-container">
                        <div className="bar-fill" style={{ width: `${result.land_cover?.classes?.total_vegetation_pct || 0}%`, background: '#22c55e' }}></div>
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                        <span style={{ color: '#38bdf8' }}>🌊 Surface Water</span>
                        <strong>{result.land_cover?.classes?.water_pct}%</strong>
                      </div>
                      <div className="bar-container">
                        <div className="bar-fill" style={{ width: `${result.land_cover?.classes?.water_pct || 0}%`, background: '#38bdf8' }}></div>
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                        <span style={{ color: '#f97316' }}>🏙️ Built-up Structures</span>
                        <strong>{result.land_cover?.classes?.built_up_pct}%</strong>
                      </div>
                      <div className="bar-container">
                        <div className="bar-fill" style={{ width: `${result.land_cover?.classes?.built_up_pct || 0}%`, background: '#f97316' }}></div>
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                        <span style={{ color: '#eab308' }}>🛣️ Roads & Infrastructure</span>
                        <strong>{result.land_cover?.classes?.roads_pct}%</strong>
                      </div>
                      <div className="bar-container">
                        <div className="bar-fill" style={{ width: `${result.land_cover?.classes?.roads_pct || 0}%`, background: '#eab308' }}></div>
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                        <span style={{ color: '#d97706' }}>🏜️ Bare Soil / Sand</span>
                        <strong>{result.land_cover?.classes?.bare_soil_pct}%</strong>
                      </div>
                      <div className="bar-container">
                        <div className="bar-fill" style={{ width: `${result.land_cover?.classes?.bare_soil_pct || 0}%`, background: '#d97706' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 2: Visual Evidence Gallery */}
            {activeTab === 'evidence' && (
              <div>
                {result.visual_evidence?.length > 0 ? (
                  <div>
                    {/* Switcher pills */}
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
                      {result.visual_evidence.map((ev, idx) => (
                        <button
                          key={idx}
                          className={`tab-btn ${selectedEvidenceIdx === idx ? 'active' : ''}`}
                          onClick={() => setSelectedEvidenceIdx(idx)}
                        >
                          <Eye size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> {ev.title}
                        </button>
                      ))}
                    </div>

                    {/* Active preview box */}
                    {result.visual_evidence[selectedEvidenceIdx] && (
                      <div className="evidence-preview-box">
                        <h4 style={{ fontSize: '15px', color: '#38bdf8', marginBottom: '6px' }}>
                          {result.visual_evidence[selectedEvidenceIdx].title}
                        </h4>
                        <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '14px' }}>
                          {result.visual_evidence[selectedEvidenceIdx].description}
                        </p>
                        <img
                          src={result.visual_evidence[selectedEvidenceIdx].url}
                          alt={result.visual_evidence[selectedEvidenceIdx].title}
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  <p style={{ color: '#64748b', fontStyle: 'italic' }}>No visual evidence required for this query.</p>
                )}
              </div>
            )}

            {/* Tab 3: Geospatial Map View */}
            {activeTab === 'map' && (
              <div>
                {result.map?.georeferenced ? (
                  <div>
                    <div style={{ display: 'flex', gap: '16px', marginBottom: '12px', fontSize: '13px', color: '#94a3b8' }}>
                      <span><strong>CRS:</strong> {result.map.crs}</span>
                      <span><strong>Footprint Center:</strong> {result.map.center?.[0]?.toFixed(5)}° N, {result.map.center?.[1]?.toFixed(5)}° E</span>
                      {result.location?.country && <span><strong>Location:</strong> {result.location.state}, {result.location.country}</span>}
                    </div>
                    <div id="leaflet-map" ref={mapRef}></div>
                  </div>
                ) : (
                  <div style={{ padding: '28px', textAlign: 'center', background: '#090e1a', borderRadius: '8px', border: '1px solid #1e293b' }}>
                    <Info size={28} color="#94a3b8" style={{ margin: '0 auto 10px' }} />
                    <h4 style={{ fontSize: '15px', color: '#cbd5e1' }}>Geographic Coordinates Unavailable</h4>
                    <p style={{ fontSize: '13px', color: '#64748b', marginTop: '6px' }}>
                      {result.map?.message || "This imagery does not contain valid georeferencing metadata (CRS/Affine bounds). In compliance with the No-Fake-Results rule, synthetic coordinates are strictly not fabricated."}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Tab 4: Auditable Execution Trace */}
            {activeTab === 'trace' && result.execution_trace && (
              <div style={{ overflowX: 'auto' }}>
                <table className="trace-table">
                  <tbody>
                    <tr>
                      <th style={{ width: '220px' }}>Selected Specialist Task</th>
                      <td><code>{result.execution_trace.task}</code></td>
                    </tr>
                    <tr>
                      <th>Active Models & Tools</th>
                      <td>
                        {result.execution_trace.models?.map(m => `${m.name} (v${m.version})`).join(' | ') || 'N/A'}
                      </td>
                    </tr>
                    <tr>
                      <th>Execution Hardware</th>
                      <td><code>{result.execution_trace.device}</code> (Hardware Aware)</td>
                    </tr>
                    <tr>
                      <th>Inference Latency</th>
                      <td>{result.execution_trace.duration_ms} ms</td>
                    </tr>
                    <tr>
                      <th>Input Characteristics</th>
                      <td>
                        {result.execution_trace.inputs?.map(i => `${i.filename} (${i.dimensions}, ${i.modality})`).join(' | ')}
                      </td>
                    </tr>
                    <tr>
                      <th>Output Integrity Status</th>
                      <td style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <CheckCircle2 size={14} /> {result.execution_trace.status_message}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
