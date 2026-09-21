// Dashboard.jsx
import { useEffect, useState, useRef } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const LABEL_META = {
  send_campaign:       { color: '#22c55e', bg: '#F0FDF4', textColor: '#15803D', text: 'Send Campaign',       action: 'Send next scheduled campaign' },
  dont_send:           { color: '#f59e0b', bg: '#FFFBEB', textColor: '#B45309', text: "Don't Send",          action: 'Wait — re-evaluate in 2 weeks' },
  no_campaign_needed:  { color: '#3b82f6', bg: '#EFF6FF', textColor: '#1D4ED8', text: 'No Campaign Needed',  action: 'NPIs, loyalty offers & seasonal only' },
  no_campaign_impact:  { color: '#ef4444', bg: '#FEF2F2', textColor: '#DC2626', text: 'No Campaign Impact',  action: 'Suppress — 90-day re-engagement' },
}

// Demo cards tell the ensemble routing story — 4 routing paths
const DEMO_CUSTOMERS = [
  {
    label: 'send_campaign',
    routing: 'ML · High Confidence',
    routingColor: '#6EE7B7', routingBg: 'rgba(110,231,183,0.12)',
    segment: 'Repeat Purchaser · Campaign Driven',
    confidence: 0.95,
    reasoning: 'Purchases consistently within 2 days of receiving a campaign. Approaching her next expected purchase window — 43 days since last order against a 68-day cycle. One of your most reliable responders.',
    signals: ['100% campaign influence rate · 8 orders', '82% email open rate · 31% click rate', '$1,368 LTV · AOV $114'],
    routingNote: 'RFC confidence 95% → ML decides automatically',
  },
  {
    label: 'no_campaign_needed',
    routing: 'Claude · Edge Case Reasoning',
    routingColor: '#93C5FD', routingBg: 'rgba(147,197,253,0.12)',
    segment: 'Loyal Buyer · Organic Purchaser',
    confidence: 0.92,
    reasoning: '14 orders and $1,035 LTV with zero campaign influence. Buys on his own schedule regardless of what you send. Protect from cadence fatigue — reserve this customer for NPIs, exclusive launches, and seasonal moments only.',
    signals: ['0% campaign influence rate · pure organic buyer', '0% email open rate — unsubscribed or ignoring', '$1,035 LTV · AOV $73'],
    routingNote: 'RFC confidence 51% → Claude reasons through the signals',
  },
  {
    label: 'no_campaign_impact',
    routing: 'ML · Flagged for Review',
    routingColor: '#FCD34D', routingBg: 'rgba(252,211,77,0.12)',
    segment: 'Occasional Buyer · Low Engagement',
    confidence: 0.71,
    reasoning: '18 campaigns received with no measurable response and a 5.6% open rate. Spend was concentrated in early 2022 and has since stalled. Suppress from regular cadence and route to a 90-day re-engagement flow before resuming.',
    signals: ['18 campaigns · 0 influenced purchases', 'Last order 8 months ago — significant drift', '$142 LTV · no recent activity'],
    routingNote: 'RFC confidence 71% → ML label, flagged for human review',
  },
  {
    label: 'dont_send',
    routing: 'Both Pipelines Agree',
    routingColor: '#CBD5E1', routingBg: 'rgba(203,213,225,0.10)',
    segment: 'Recent Purchaser · Post-Purchase Window',
    confidence: 0.88,
    reasoning: 'Purchased 6 days ago. Both the ML model and Claude agree — sending now risks fatigue and unsubscribe. Strong campaign-driven buyer history means she is a good candidate again once the purchase cycle resets in two weeks.',
    signals: ['6 days since last order — post-purchase window', 'ML + Claude both predict dont_send', 'High campaign influence rate — re-queue in 2 weeks'],
    routingNote: '74% of customers: both pipelines reach the same label',
  },
]

function StatCard({ label, value, sub }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <div className="text-[10px] font-bold tracking-widest uppercase text-gray-400 mb-1.5">{label}</div>
      <div className="text-[30px] font-bold text-brand-navy leading-none tracking-tight">{value ?? '—'}</div>
      <div className="text-[11px] text-gray-400 mt-1">{sub}</div>
    </div>
  )
}

function ScorecardRow({ labelKey, count, pct, ltv }) {
  const meta = LABEL_META[labelKey]
  if (!meta) return null
  return (
    <div className="flex items-center px-3.5 py-3 rounded-lg"
      style={{ background: meta.bg, borderLeft: `3px solid ${meta.color}` }}>
      <div className="flex-1 min-w-0">
        <div className="text-[12px] font-bold tracking-wide uppercase" style={{ color: meta.textColor }}>{meta.text}</div>
        <div className="text-[11px] text-gray-500 mt-0.5">{meta.action}</div>
      </div>
      <div className="text-right flex-shrink-0 ml-4">
        <div className="text-[20px] font-bold text-brand-navy leading-none">{count ?? '—'}</div>
        <div className="text-[10px] text-gray-400 mt-0.5">
          {pct != null ? `${Number(pct).toFixed(1)}%` : '—'} · ${ltv != null ? Math.round(ltv) : '—'} avg LTV
        </div>
      </div>
    </div>
  )
}

function DemoLoop() {
  const [idx, setIdx] = useState(0)
  const timerRef = useRef(null)
  useEffect(() => {
    timerRef.current = setInterval(() => setIdx(i => (i + 1) % DEMO_CUSTOMERS.length), 6000)
    return () => clearInterval(timerRef.current)
  }, [])
  const c = DEMO_CUSTOMERS[idx]
  const meta = LABEL_META[c.label]
  return (
    <div className="rounded-xl p-5 flex flex-col" style={{ background: '#1D3251', minHeight: 300 }}>
      <div className="flex justify-between items-center mb-4">
        <div>
          <div className="text-[11px] font-bold tracking-widest uppercase" style={{ color: 'rgba(255,255,255,0.38)' }}>Agent Reasoning</div>
          <div className="text-[13px] font-semibold text-white mt-0.5">Live Demo — auto-cycles every 6s</div>
        </div>
        <div className="text-[10px] font-bold tracking-wide px-2 py-1 rounded border"
          style={{ background: 'rgba(194,104,32,0.18)', borderColor: '#C26820', color: '#D4884A' }}>LIVE</div>
      </div>
      <div className="flex-1 rounded-lg p-4" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.09)' }}>
        {/* Routing badge */}
        <div className="inline-flex items-center px-2.5 py-1 rounded-full mb-3 text-[10px] font-bold tracking-wide uppercase"
          style={{ background: c.routingBg, color: c.routingColor, border: `1px solid ${c.routingColor}50` }}>
          {c.routing}
        </div>
        <div className="text-[10px] tracking-widest uppercase mb-2" style={{ color: 'rgba(255,255,255,0.38)', fontFamily: 'monospace' }}>{c.segment}</div>
        <div className="flex items-center gap-2.5 mb-3">
          <span className="text-[10px] font-bold tracking-wide uppercase px-2.5 py-1 rounded text-white" style={{ background: meta.color }}>{meta.text}</span>
          <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.45)' }}>Confidence {c.confidence.toFixed(2)}</span>
        </div>
        <div className="h-[3px] rounded-full mb-3.5" style={{ background: 'rgba(255,255,255,0.08)' }}>
          <div className="h-[3px] rounded-full transition-all duration-700" style={{ background: meta.color, width: `${c.confidence * 100}%` }} />
        </div>
        <p className="text-[13px] leading-relaxed mb-3 italic" style={{ color: 'rgba(255,255,255,0.82)', fontFamily: 'Georgia, serif', margin: '0 0 12px' }}>
          "{c.reasoning}"
        </p>
        <div className="flex flex-col gap-1 mb-3">
          {c.signals.map((s, i) => (
            <div key={i} className="text-[11px]" style={{ color: 'rgba(255,255,255,0.48)' }}>
              <span className="mr-1.5" style={{ color: '#C26820' }}>•</span>{s}
            </div>
          ))}
        </div>
        <div className="text-[10px] italic pt-2" style={{ color: 'rgba(255,255,255,0.28)', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          {c.routingNote}
        </div>
      </div>
      <div className="flex justify-center gap-2 mt-3.5">
        {DEMO_CUSTOMERS.map((_, i) => (
          <button key={i} onClick={() => { setIdx(i); clearInterval(timerRef.current); timerRef.current = setInterval(() => setIdx(n => (n + 1) % DEMO_CUSTOMERS.length), 6000) }}
            className="w-2 h-2 rounded-full transition-colors duration-300"
            style={{ background: i === idx ? '#C26820' : 'rgba(255,255,255,0.18)' }} />
        ))}
      </div>
    </div>
  )
}

const REVENUE_BARS = [
  {h:20},{h:28},{h:25},{h:32},{h:30},{h:40,s:true},{h:38,s:true},{h:45},{h:42},{h:35},{h:55,hol:true},{h:60,hol:true},
  {h:38},{h:42},{h:55},{h:62},{h:58},{h:68,s:true},{h:65,s:true},{h:55},{h:60},{h:52},{h:78,hol:true},{h:82,hol:true},
  {h:55},{h:52},{h:70},{h:65},{h:72},{h:80,s:true},{h:85,s:true},{h:72},{h:75},{h:68},{h:92,hol:true},{h:100,hol:true},
]

export default function Dashboard() {
  const [overview,  setOverview]  = useState(null)
  const [scorecard, setScorecard] = useState([])
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const [ovRes, scRes] = await Promise.all([
          axios.get(`${API}/analytics/overview`),
          axios.get(`${API}/recommendations/scorecard`),
        ])
        setOverview(ovRes.data)
        // API returns { scorecard: [...] }
        const sc = scRes.data?.scorecard ?? scRes.data ?? []
        setScorecard(Array.isArray(sc) ? sc : [])
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div className="flex items-center justify-center h-64 text-gray-400 text-sm">Loading dashboard…</div>
  if (error)   return <div className="p-6 text-sm text-red-500 font-mono">API error: {error}<br/>Make sure the API is running on {API}</div>

  // Scorecard rows keyed by label
  const sc = {}
  scorecard.forEach(row => { sc[row.label] = row })

  const totalCustomers = overview?.total_customers ?? overview?.customers
  const totalOrders    = overview?.total_orders
  const totalRevenue   = overview?.total_revenue
    ? `$${(Number(overview.total_revenue) / 1000).toFixed(1)}K` : '—'
  const totalCampaigns = overview?.total_campaigns

  return (
    <div>
      <div className="flex justify-between items-baseline mb-7">
        <div>
          <h1 className="text-[22px] font-bold text-brand-navy tracking-tight m-0">Campaign Intelligence Dashboard</h1>
          <p className="text-[13px] text-gray-500 mt-1">
            {totalCustomers ? `${totalCustomers} customers` : ''} · Powered by AIIR Framework · claude-sonnet-4-6
          </p>
        </div>
        <div className="text-[12px] font-semibold px-4 py-2 rounded-lg bg-brand-navy text-white cursor-default">Run Agent ▶</div>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-7">
        <StatCard label="Total Customers" value={totalCustomers?.toLocaleString?.()} sub="3yr window" />
        <StatCard label="Total Orders"    value={totalOrders?.toLocaleString?.()}    sub="3yr window" />
        <StatCard label="Total Revenue"   value={totalRevenue}                        sub="paid + pending" />
        <StatCard label="Campaigns Sent"  value={totalCampaigns?.toLocaleString?.()}  sub="18-month window" />
      </div>

      <div className="grid grid-cols-2 gap-5 mb-6">
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <div className="text-[11px] font-bold tracking-widest uppercase text-gray-400 mb-4">Recommendation Scorecard</div>
          <div className="flex flex-col gap-2.5">
            {['send_campaign','dont_send','no_campaign_needed','no_campaign_impact'].map(key => {
              const row = sc[key] || {}
              const total = scorecard.reduce((s, r) => s + (r.customers ?? 0), 0)
              return (
                <ScorecardRow key={key} labelKey={key}
                  count={row.customers}
                  pct={row.customers && total ? (row.customers / total * 100) : null}
                  ltv={row.avg_ltv} />
              )
            })}
          </div>
        </div>
        <DemoLoop />
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <div className="text-[11px] font-bold tracking-widest uppercase text-gray-400 mb-4">Monthly Revenue Trend · Jan 2022 – Dec 2024</div>
        <div className="flex items-end gap-1" style={{ height: 80 }}>
          {REVENUE_BARS.map((b, i) => (
            <div key={i} style={{ flex: 1, borderRadius: '2px 2px 0 0', height: `${b.h}%`,
              background: b.hol ? '#C26820' : b.s ? '#C2CBE8' : '#E4E6EE',
              opacity: b.hol && i < 12 ? 0.7 : b.hol && i < 24 ? 0.8 : 1 }} />
          ))}
        </div>
        <div className="flex justify-between mt-1.5 text-[10px] text-gray-400">
          <span>Jan 2022</span>
          <span className="font-semibold" style={{ color: '#C26820' }}>▲ Nov/Dec holiday peaks</span>
          <span className="font-semibold" style={{ color: '#C2CBE8' }}>▲ Summer cold brew</span>
          <span>Dec 2024</span>
        </div>
      </div>
    </div>
  )
}
