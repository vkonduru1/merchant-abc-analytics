// Analytics.jsx
import { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Fallback segment data (from case_04 backtest results)
const FALLBACK_SEGMENTS = [
  { segment: 'campaign_3_5',          customers: 174, avg_campaigns: 3.8, avg_ltv: 276, influence_pct: 92,  bar_pct: 100 },
  { segment: 'campaign_1_2',          customers: 100, avg_campaigns: 1.4, avg_ltv: 198, influence_pct: 78,  bar_pct: 72  },
  { segment: 'no_campaign_influence', customers: 185, avg_campaigns: null, avg_ltv: 142, influence_pct: 0,  bar_pct: 51  },
  { segment: 'campaign_gt5',          customers: null, avg_campaigns: 7.2, avg_ltv: 98, influence_pct: 61, bar_pct: 36  },
  { segment: 'never_purchased',       customers: 37,  avg_campaigns: 22.4, avg_ltv: 0,  influence_pct: null, bar_pct: 2 },
]

function AIIRStrip() {
  const items = [
    {
      letter: 'A', label: 'Analysis',
      desc: 'Raw event data examined. 30-day attribution windows. Campaign influence measured per order.',
      table: 'customer_events + campaign_attribution',
    },
    {
      letter: 'I', label: 'Insight',
      desc: 'First-level features computed from Analysis. 39 fields: recency, frequency, spend windows, email engagement rates, influence rate.',
      table: 'customer_derivatives',
    },
    {
      letter: 'I', label: 'Interpretation',
      desc: 'Third-level derivatives — the significant predictors that feed the classifier. Attribution segment, campaign influence rate, avg campaigns before purchase.',
      table: 'distilled signals Claude reasons from',
    },
    {
      letter: 'R', label: 'Recommendation',
      desc: 'Confidence-based ensemble: RFC classifies at high confidence (≥0.80), Claude reasons through edge cases (<0.60), middle-band predictions flagged for review. 4 labels + written reasoning per customer.',
      table: 'customer_recommendations',
    },
  ]
  return (
    <div className="rounded-xl mb-6 overflow-hidden" style={{ background: '#1D3251' }}>
      {/* Pre-AIIR note */}
      <div className="px-5 py-3 text-[12px]" style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.04)' }}>
        <span className="font-bold tracking-widest uppercase mr-2" style={{ color: '#D4884A' }}>Pre-AIIR (ETL)</span>
        <span style={{ color: 'rgba(255,255,255,0.75)' }}>Identity resolution — Shopify + Klaviyo joined via email · <span style={{ fontFamily: 'monospace', color: 'rgba(255,255,255,0.55)' }}>customer_identity_map · customer_events</span> · This is infrastructure, not the framework</span>
      </div>
      {/* AIIR columns */}
      <div className="flex p-4">
        {items.map((item, i) => (
          <div key={i} className="flex-1 px-4"
            style={{
              borderRight: i < items.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none',
              paddingLeft: i === 0 ? 0 : undefined,
            }}>
            <div className="text-[10px] font-bold tracking-widest uppercase mb-1.5" style={{ color: '#D4884A' }}>
              {item.letter} — {item.label}
            </div>
            <div className="text-[12px] leading-relaxed mb-2" style={{ color: 'rgba(255,255,255,0.65)' }}>
              {item.desc}
            </div>
            <div className="text-[10px] font-mono" style={{ color: 'rgba(255,255,255,0.3)' }}>
              {item.table}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function SegmentBar({ segment, pct, ltv, customers, maxLtv }) {
  const w = maxLtv > 0 ? (ltv / maxLtv) * 100 : pct
  const highlight = segment === 'campaign_3_5'
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1">
        <div className="text-[12px] font-semibold" style={{ color: highlight ? '#17274C' : '#374151', fontFamily: 'monospace' }}>
          {segment}
        </div>
        <div className="text-[12px] font-bold" style={{ color: highlight ? '#C26820' : '#374151' }}>
          ${ltv?.toLocaleString() ?? 0}
        </div>
      </div>
      <div className="rounded-sm" style={{ background: '#F3F4F6', height: 8 }}>
        <div className="rounded-sm h-2 transition-all duration-500"
          style={{ background: highlight ? '#C26820' : '#C2CBE8', width: `${w}%` }} />
      </div>
      {customers && (
        <div className="text-[10px] text-gray-400 mt-0.5">{customers.toLocaleString()} customers</div>
      )}
    </div>
  )
}

export default function Analytics() {
  const [attribution, setAttribution] = useState([])
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        const res = await axios.get(`${API}/analytics/attribution-summary`)
        setAttribution(res.data ?? [])
      } catch (e) {
        setError(e.message)
        // Use fallback silently — the chart still renders
        setAttribution([])
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const segments = attribution.length ? attribution : FALLBACK_SEGMENTS
  const maxLtv = Math.max(...segments.map(s => s.avg_ltv ?? s.avg_lifetime_revenue ?? 0), 1)

  // Normalise field names from API
  const normalised = segments.map(s => ({
    segment:       s.attribution_segment ?? s.segment ?? '—',
    customers:     s.customer_count ?? s.customers ?? null,
    avg_campaigns: s.avg_campaigns_before_purchase ?? s.avg_campaigns ?? null,
    avg_ltv:       s.avg_lifetime_revenue ?? s.avg_ltv ?? 0,
    influence_pct: s.avg_influence_rate_pct ?? s.influence_pct ?? null,
    bar_pct:       s.bar_pct,
  }))

  if (loading) return <div className="flex items-center justify-center h-64 text-gray-400 text-sm">Loading attribution…</div>

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-[22px] font-bold text-brand-navy tracking-tight m-0">Campaign Attribution</h1>
        <p className="text-[13px] text-gray-500 mt-1">
          AIIR Framework — Analysis · Insight · Interpretation · Recommendation
          {error && <span className="ml-2 text-amber-500">(showing cached data)</span>}
        </p>
      </div>

      {/* AIIR explainer strip */}
      <AIIRStrip />

      {/* Chart + table */}
      <div className="grid grid-cols-2 gap-5 mb-5">

        {/* Bar chart */}
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <div className="text-[11px] font-bold tracking-widest uppercase text-gray-400 mb-5">Avg LTV by Attribution Segment</div>
          <div className="flex flex-col gap-4">
            {normalised.map(s => (
              <SegmentBar
                key={s.segment}
                segment={s.segment}
                ltv={s.avg_ltv}
                customers={s.customers}
                pct={s.bar_pct ?? ((s.avg_ltv / maxLtv) * 100)}
                maxLtv={maxLtv}
              />
            ))}
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-[11px] text-gray-400 italic"
            style={{ fontFamily: 'Georgia, serif' }}>
            campaign_3_5 customers generate {normalised[0]?.avg_ltv && normalised[2]?.avg_ltv
              ? `${(normalised[0].avg_ltv / Math.max(normalised[2].avg_ltv, 1)).toFixed(1)}x`
              : '2.4x'} more LTV than those uninfluenced by campaigns.
          </div>
        </div>

        {/* Segment table */}
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="grid gap-2 px-4 py-2.5 bg-gray-50 border-b border-gray-200 text-[10px] font-bold tracking-widest uppercase text-gray-400"
            style={{ gridTemplateColumns: '1.5fr 0.7fr 1fr 0.8fr 0.8fr' }}>
            <div>Segment</div>
            <div>Customers</div>
            <div>Avg Campaigns</div>
            <div>Avg LTV</div>
            <div>Influence %</div>
          </div>

          {normalised.map(s => (
            <div key={s.segment}
              className="grid gap-2 px-4 py-3 border-b border-gray-50 last:border-0 items-center"
              style={{ gridTemplateColumns: '1.5fr 0.7fr 1fr 0.8fr 0.8fr' }}>
              <div className="text-[12px] font-semibold font-mono"
                style={{ color: s.segment === 'campaign_3_5' ? '#17274C' : s.avg_ltv === 0 ? '#9CA3AF' : '#374151' }}>
                {s.segment}
              </div>
              <div className="text-[13px] text-gray-700">
                {s.customers?.toLocaleString() ?? <span className="text-gray-400">variable</span>}
              </div>
              <div className="text-[13px] text-gray-700">
                {s.avg_campaigns != null ? s.avg_campaigns : <span className="text-gray-400">—</span>}
              </div>
              <div className="text-[13px] font-bold"
                style={{ color: s.avg_ltv > 0 ? (s.segment === 'campaign_3_5' ? '#C26820' : '#374151') : '#9CA3AF' }}>
                ${(s.avg_ltv ?? 0).toLocaleString()}
              </div>
              <div className="text-[13px]"
                style={{ color: s.influence_pct != null ? '#374151' : '#9CA3AF' }}>
                {s.influence_pct != null ? `${s.influence_pct}%` : '—'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key insight callout */}
      <div className="rounded-xl p-4 flex gap-4 items-start"
        style={{ background: '#FBF8F2', border: '1px solid #E4D5C2' }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: '#C26820' }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
        </div>
        <div>
          <div className="text-[12px] font-bold text-brand-navy mb-1">Key Business Insight</div>
          <div className="text-[13px] text-gray-600 leading-relaxed" style={{ fontFamily: 'Georgia, serif' }}>
            The <strong>97 customers</strong> classified as <em>no_campaign_impact</em> ($40 avg LTV) are consuming
            campaign budget with no return — suppressing them improves deliverability for the 399 customers who do
            respond. The <strong>129 no_campaign_needed</strong> brand-loyal buyers ($131 avg LTV) should receive
            NPIs and loyalty offers only, not the regular newsletter cadence.
          </div>
        </div>
      </div>
    </div>
  )
}
