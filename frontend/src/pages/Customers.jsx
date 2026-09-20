// Customers.jsx
import { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const LABEL_META = {
  send_campaign:       { color: '#22c55e', bg: '#dcfce7', textColor: '#15803D', text: 'Send Campaign'      },
  dont_send:           { color: '#f59e0b', bg: '#fef3c7', textColor: '#B45309', text: "Don't Send"         },
  no_campaign_needed:  { color: '#3b82f6', bg: '#dbeafe', textColor: '#1D4ED8', text: 'No Campaign Needed' },
  no_campaign_impact:  { color: '#ef4444', bg: '#fee2e2', textColor: '#DC2626', text: 'No Campaign Impact' },
}

const FILTERS = [
  { id: 'all',               label: 'All' },
  { id: 'send_campaign',     label: 'Send Campaign' },
  { id: 'dont_send',         label: "Don't Send" },
  { id: 'no_campaign_needed',label: 'No Campaign Needed' },
  { id: 'no_campaign_impact',label: 'No Campaign Impact' },
]

function LabelBadge({ label }) {
  const meta = LABEL_META[label]
  if (!meta) return <span className="text-[11px] text-gray-400 font-mono">{label ?? '—'}</span>
  return (
    <span className="text-[10px] font-bold tracking-wide uppercase px-2 py-1 rounded"
      style={{ background: meta.bg, color: meta.textColor }}>{meta.text}</span>
  )
}

function DrillDown({ c }) {
  return (
    <div className="px-4 py-3.5 border-t border-dashed" style={{ background: '#F8FEF9', borderColor: '#D1FAE5' }}>
      <div className="text-[11px] font-bold tracking-widest uppercase mb-2" style={{ color: '#15803D' }}>Claude's Reasoning</div>
      <p className="text-[13px] leading-relaxed mb-2.5 italic"
        style={{ fontFamily: 'Georgia, serif', color: '#374151', margin: '0 0 10px' }}>
        "{c.recommendation_reasoning || c.reasoning || 'No reasoning recorded.'}"
      </p>
      <div className="flex flex-wrap gap-4 text-[11px] text-gray-500">
        {c.avg_influence_rate_pct != null && <span><span className="mr-1" style={{color:'#C85510'}}>•</span>{c.avg_influence_rate_pct}% influence rate</span>}
        {c.email_open_rate_pct    != null && <span><span className="mr-1" style={{color:'#C85510'}}>•</span>{c.email_open_rate_pct}% open rate</span>}
        {c.total_orders           != null && <span><span className="mr-1" style={{color:'#C85510'}}>•</span>{c.total_orders} orders</span>}
        {c.lifetime_revenue       != null && <span><span className="mr-1" style={{color:'#C85510'}}>•</span>${Number(c.lifetime_revenue).toFixed(0)} LTV</span>}
        {c.confidence_score       != null && <span><span className="mr-1" style={{color:'#C85510'}}>•</span>conf {Number(c.confidence_score).toFixed(2)}</span>}
      </div>
    </div>
  )
}

export default function Customers() {
  const [customers,  setCustomers]  = useState([])
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [filter,     setFilter]     = useState('all')
  const [sortBy,     setSortBy]     = useState('ltv')
  const [expandedId, setExpandedId] = useState(null)
  const [page,       setPage]       = useState(1)
  const PAGE_SIZE = 20

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        // Fetch recommendations (label + reasoning) and customers (behavioral metrics) in parallel
        const [recRes, custRes] = await Promise.allSettled([
          axios.get(`${API}/recommendations/`, { params: { limit: 500 } }),
          axios.get(`${API}/customers/`,       { params: { limit: 500 } }),
        ])

        const recList  = recRes.status  === 'fulfilled' ? (recRes.value.data?.recommendations  ?? recRes.value.data  ?? []) : []
        const custList = custRes.status === 'fulfilled' ? (custRes.value.data?.customers        ?? custRes.value.data ?? []) : []

        // Build customer lookup map for behavioral metrics
        const custMap = {}
        if (Array.isArray(custList)) {
          custList.forEach(c => { custMap[c.customer_id ?? c.id] = c })
        }

        // Merge: start with recommendations, overlay customer fields
        let merged = []
        if (Array.isArray(recList) && recList.length) {
          merged = recList.map(r => ({
            ...custMap[r.customer_id] ?? {},
            ...r,
          }))
        } else if (Array.isArray(custList) && custList.length) {
          merged = custList
        }

        setCustomers(merged)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  // Normalise label field — could be recommendation or label
  const normalise = c => ({
    ...c,
    _label: c.recommendation ?? c.label ?? c.recommendation_label ?? '',
    _id:    c.customer_id ?? c.id ?? Math.random(),
  })

  const rows = customers.map(normalise)
  const filtered = rows
    .filter(c => filter === 'all' || c._label === filter)
    .sort((a, b) => {
      if (sortBy === 'ltv')     return (b.lifetime_revenue ?? b.total_revenue ?? 0) - (a.lifetime_revenue ?? a.total_revenue ?? 0)
      if (sortBy === 'orders')  return (b.total_orders ?? 0) - (a.total_orders ?? 0)
      if (sortBy === 'recency') return (a.days_since_last_order ?? 9999) - (b.days_since_last_order ?? 9999)
      return 0
    })

  const paged = filtered.slice(0, page * PAGE_SIZE)

  if (loading) return <div className="flex items-center justify-center h-64 text-gray-400 text-sm">Loading customers…</div>
  if (error)   return <div className="p-6 text-sm text-red-500 font-mono">API error: {error}</div>

  return (
    <div>
      <div className="flex justify-between items-baseline mb-6">
        <h1 className="text-[22px] font-bold text-brand-navy tracking-tight m-0">Customers</h1>
        <select value={sortBy} onChange={e => setSortBy(e.target.value)}
          className="text-[12px] text-gray-500 bg-gray-100 rounded-lg px-3 py-1.5 border-0 outline-none cursor-pointer">
          <option value="ltv">Sort: LTV ↓</option>
          <option value="orders">Sort: Orders ↓</option>
          <option value="recency">Sort: Recency ↑</option>
        </select>
      </div>

      {/* Filter pills */}
      <div className="flex gap-2 flex-wrap mb-5">
        {FILTERS.map(f => (
          <button key={f.id} onClick={() => { setFilter(f.id); setPage(1) }}
            className="text-[11px] font-semibold px-3 py-1.5 rounded-full transition-colors"
            style={filter === f.id
              ? { background: '#17274C', color: '#fff', border: '1px solid #17274C' }
              : { background: 'white', color: '#6B7280', border: '1px solid #E5E7EB' }}>
            {f.label}
            {f.id !== 'all' && (
              <span className="ml-1.5 opacity-60">{rows.filter(c => c._label === f.id).length}</span>
            )}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="grid gap-2 px-4 py-2.5 bg-gray-50 border-b border-gray-200 text-[10px] font-bold tracking-widest uppercase text-gray-400"
          style={{ gridTemplateColumns: '2fr 1.4fr 0.7fr 0.8fr 0.7fr 0.7fr' }}>
          <div>Customer</div><div>Label</div><div>Orders</div><div>LTV</div><div>Recency</div><div>Open %</div>
        </div>

        {paged.length === 0 && (
          <div className="py-16 text-center text-sm text-gray-400 italic">No customers match this filter.</div>
        )}

        {paged.map(c => {
          const expanded = expandedId === c._id
          const ltv = c.lifetime_revenue ?? c.total_revenue
          const openRate = c.email_open_rate_pct ?? c.open_rate
          const name = c.first_name && c.last_name
            ? `${c.first_name} ${c.last_name[0]}.`
            : c.email?.split('@')[0] ?? c.customer_id ?? '—'
          return (
            <div key={c._id} className="border-b border-gray-100 last:border-0">
              <div className="grid gap-2 px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors items-center"
                style={{ gridTemplateColumns: '2fr 1.4fr 0.7fr 0.8fr 0.7fr 0.7fr',
                  background: expanded ? (LABEL_META[c._label]?.bg ?? '#F9FAFB') : undefined }}
                onClick={() => setExpandedId(prev => prev === c._id ? null : c._id)}>
                <div>
                  <div className="text-[13px] font-semibold text-brand-navy">{name}</div>
                  <div className="text-[11px] text-gray-400 font-mono">{c.customer_id}</div>
                </div>
                <div><LabelBadge label={c._label} /></div>
                <div className="text-[13px] text-gray-700">{c.total_orders ?? '—'}</div>
                <div className="text-[13px] text-gray-700">
                  {ltv != null ? `$${Number(ltv).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '—'}
                </div>
                <div className="text-[13px] text-gray-700">
                  {c.days_since_last_order != null ? `${c.days_since_last_order}d` : '—'}
                </div>
                <div className="text-[13px] text-gray-700">
                  {openRate != null ? `${openRate}%` : '—'}
                </div>
              </div>
              {expanded && <DrillDown c={c} />}
            </div>
          )
        })}
      </div>

      <div className="mt-3 text-[12px] text-gray-400 text-center">
        Showing {paged.length} of {filtered.length} · Click any row to expand Claude's reasoning
        {paged.length < filtered.length && (
          <button onClick={() => setPage(p => p + 1)} className="ml-3 font-semibold hover:underline" style={{ color: '#C85510' }}>
            Load more
          </button>
        )}
      </div>
    </div>
  )
}
