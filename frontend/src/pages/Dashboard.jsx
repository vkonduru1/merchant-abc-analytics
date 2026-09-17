import { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function StatCard({ label, value, sub }) {
  return (
    <div className="bg-white rounded-lg p-5 shadow-sm border border-gray-100">
      <div className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-1">{label}</div>
      <div className="text-3xl font-bold text-brand-navy">{value ?? '—'}</div>
      {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
    </div>
  )
}

export default function Dashboard() {
  const [overview, setOverview] = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)

  useEffect(() => {
    axios.get(`${API}/analytics/overview`)
      .then(r => { setOverview(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <p className="text-gray-400 text-sm">Loading dashboard…</p>
  if (error)   return <p className="text-red-500 text-sm">API error: {error}. Is the backend running?</p>

  return (
    <div>
      <h1 className="text-2xl font-bold text-brand-navy mb-6">Overview</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Customers"      value={overview?.total_customers?.toLocaleString()} />
        <StatCard label="Total Orders"   value={overview?.total_orders?.toLocaleString()} />
        <StatCard label="Total Revenue"  value={overview?.total_revenue
          ? `$${Number(overview.total_revenue).toLocaleString('en-US', {minimumFractionDigits:2})}`
          : '—'} />
        <StatCard label="Campaigns Sent" value={overview?.total_campaigns?.toLocaleString()} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg p-5 shadow-sm border border-gray-100">
          <div className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-3">
            Avg Campaigns Before Purchase
          </div>
          <div className="text-4xl font-bold text-brand-orange">
            {overview?.avg_campaigns_before_purchase ?? '—'}
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Average number of email campaigns sent to a customer before they make a purchase.
          </p>
        </div>

        <div className="bg-white rounded-lg p-5 shadow-sm border border-gray-100">
          <div className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-3">
            Data Pipeline Status
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Emails tracked</span>
              <span className="font-medium">{overview?.total_emails_sent?.toLocaleString() ?? '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Purchase events</span>
              <span className="font-medium">{overview?.total_purchase_events?.toLocaleString() ?? '—'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
