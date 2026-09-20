// Campaigns.jsx
import { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const TYPE_META = {
  seasonal:     { bg: '#FEF3C7', color: '#B45309' },
  promo:        { bg: '#FEE2E2', color: '#DC2626' },
  reactivation: { bg: '#F3E8FF', color: '#7C3AED' },
  newsletter:   { bg: '#E0F2FE', color: '#0369A1' },
  welcome:      { bg: '#D1FAE5', color: '#065F46' },
}

function TypeBadge({ type }) {
  const key = Object.keys(TYPE_META).find(k => (type || '').toLowerCase().includes(k))
  const meta = TYPE_META[key] || { bg: '#F3F4F6', color: '#6B7280' }
  return (
    <span className="text-[10px] font-semibold px-2 py-0.5 rounded"
      style={{ background: meta.bg, color: meta.color }}>
      {type || '—'}
    </span>
  )
}

function SummaryPill({ label, value, accent }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl px-4 py-2.5 text-center">
      <div className="text-[18px] font-bold leading-none" style={{ color: accent ?? '#17274C' }}>{value}</div>
      <div className="text-[10px] font-bold tracking-widest uppercase text-gray-400 mt-1">{label}</div>
    </div>
  )
}

export default function Campaigns() {
  const [campaigns, setCampaigns] = useState([])
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState(null)
  const [sortBy,    setSortBy]    = useState('date')

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        const res = await axios.get(`${API}/analytics/campaign-performance`)
        // API returns { campaigns: [...] }
        const list = res.data?.campaigns ?? res.data ?? []
        setCampaigns(Array.isArray(list) ? list : [])
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const sorted = [...campaigns].sort((a, b) => {
    if (sortBy === 'date')    return new Date(b.send_time ?? b.send_date ?? 0) - new Date(a.send_time ?? a.send_date ?? 0)
    if (sortBy === 'open')    return (b['total_opens%'] ?? b.open_rate ?? 0) - (a['total_opens%'] ?? a.open_rate ?? 0)
    if (sortBy === 'revenue') return (b.influenced_revenue ?? b.total_revenue ?? 0) - (a.influenced_revenue ?? a.total_revenue ?? 0)
    return 0
  })

  const totalDelivered       = campaigns.reduce((s, c) => s + (c.total_delivered ?? 0), 0)
  const totalInfluencedOrders   = campaigns.reduce((s, c) => s + (c.influenced_orders ?? 0), 0)
  const totalInfluencedRevenue  = campaigns.reduce((s, c) => s + (c.influenced_revenue ?? c.total_revenue ?? 0), 0)
  const avgOpen = campaigns.length
    ? (campaigns.reduce((s, c) => s + (c['total_opens%'] ?? c.open_rate ?? 0), 0) / campaigns.length).toFixed(1)
    : '—'

  if (loading) return <div className="flex items-center justify-center h-64 text-gray-400 text-sm">Loading campaigns…</div>
  if (error)   return <div className="p-6 text-sm text-red-500 font-mono">API error: {error}</div>

  return (
    <div>
      <div className="flex justify-between items-baseline mb-6">
        <h1 className="text-[22px] font-bold text-brand-navy tracking-tight m-0">Campaigns</h1>
        <div className="text-[12px] text-gray-400">{campaigns.length} campaigns · 18-month window</div>
      </div>

      {/* Summary pills */}
      <div className="flex gap-3 mb-6">
        <SummaryPill label="Emails Delivered"   value={totalDelivered.toLocaleString()} />
        <SummaryPill label="Avg Open Rate"      value={`${avgOpen}%`} accent="#C85510" />
        <SummaryPill label="Influenced Orders"  value={totalInfluencedOrders.toLocaleString()} />
        <SummaryPill label="Influenced Revenue" value={
          totalInfluencedRevenue >= 1000
            ? `$${(totalInfluencedRevenue / 1000).toFixed(1)}K`
            : `$${Math.round(totalInfluencedRevenue).toLocaleString()}`
        } />
      </div>

      {/* Sort */}
      <div className="flex justify-end mb-3">
        <select value={sortBy} onChange={e => setSortBy(e.target.value)}
          className="text-[12px] text-gray-500 bg-gray-100 rounded-lg px-3 py-1.5 border-0 outline-none cursor-pointer">
          <option value="date">Sort: Date ↓</option>
          <option value="open">Sort: Open % ↓</option>
          <option value="revenue">Sort: Revenue ↓</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="grid gap-2 px-4 py-2.5 bg-gray-50 border-b border-gray-200 text-[10px] font-bold tracking-widest uppercase text-gray-400"
          style={{ gridTemplateColumns: '3fr 1fr 1fr 0.8fr 0.8fr 0.8fr 1.1fr' }}>
          <div>Campaign</div><div>Type</div><div>Date</div>
          <div>Sent</div><div>Open %</div><div>Click %</div><div>Inf. Revenue</div>
        </div>

        {sorted.length === 0 && (
          <div className="py-16 text-center text-sm text-gray-400 italic">No campaign data available.</div>
        )}

        {sorted.map((c, i) => {
          const openRate  = c['total_opens%'] ?? c.open_rate
          const clickRate = c['total_clicks%'] ?? c.click_rate
          const sent      = c.total_delivered ?? c.recipients
          const revenue   = c.influenced_revenue ?? c.total_revenue
          const date      = c.send_time ?? c.send_date

          return (
            <div key={c.campaign_id ?? i}
              className="grid gap-2 px-4 py-3 border-b border-gray-50 last:border-0 hover:bg-gray-50 transition-colors items-center"
              style={{ gridTemplateColumns: '3fr 1fr 1fr 0.8fr 0.8fr 0.8fr 1.1fr' }}>
              <div>
                <div className="text-[13px] font-medium text-brand-navy leading-snug">
                  {c.campaign_name ?? `Campaign ${c.campaign_id}`}
                </div>
                {c.discount_code
                  ? <div className="text-[10px] font-semibold mt-0.5" style={{ color: '#C85510' }}>{c.discount_code}</div>
                  : <div className="text-[10px] text-gray-400 mt-0.5">no discount code</div>}
              </div>
              <div><TypeBadge type={c.campaign_type} /></div>
              <div className="text-[12px] text-gray-500">
                {date ? new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '—'}
              </div>
              <div className="text-[13px] text-gray-700">{sent?.toLocaleString() ?? '—'}</div>
              <div className="text-[13px] font-semibold" style={{ color: (openRate ?? 0) >= 40 ? '#17274C' : '#6B7280' }}>
                {openRate != null ? `${Number(openRate).toFixed(1)}%` : '—'}
              </div>
              <div className="text-[13px] text-gray-700">
                {clickRate != null ? `${Number(clickRate).toFixed(1)}%` : '—'}
              </div>
              <div className="text-[13px] font-semibold" style={{ color: (revenue ?? 0) > 0 ? '#17274C' : '#9CA3AF' }}>
                {revenue != null ? (revenue > 0 ? `$${Number(revenue).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '$0') : '—'}
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-3 text-[12px] text-gray-400 text-center">
        {sorted.length} campaigns · Influenced revenue = orders within 30-day post-campaign window
      </div>
    </div>
  )
}
