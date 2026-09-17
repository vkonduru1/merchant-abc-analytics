// Analytics.jsx
import { useEffect, useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const SEGMENT_LABELS = {
  no_campaign_influence: 'No Campaign Influence',
  campaign_1_2: '1–2 Campaigns',
  campaign_3_5: '3–5 Campaigns',
  campaign_gt5: '> 5 Campaigns',
  never_purchased: 'Never Purchased',
}

export default function Analytics() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get(`${API}/analytics/attribution-summary`)
      .then(r => { setData(r.data.attribution_segments); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <p className="text-gray-400 text-sm">Loading…</p>
  if (!data.length) return (
    <div>
      <h1 className="text-2xl font-bold text-brand-navy mb-4">Campaign Attribution</h1>
      <p className="text-gray-400 text-sm">No attribution data yet. Run the transformation pipeline first.</p>
    </div>
  )

  const chartData = data.map(d => ({
    name: SEGMENT_LABELS[d.attribution_segment] || d.attribution_segment,
    revenue: parseFloat(d.avg_lifetime_revenue || 0),
    customers: parseInt(d.customer_count || 0),
  }))

  return (
    <div>
      <h1 className="text-2xl font-bold text-brand-navy mb-6">Campaign Attribution</h1>
      <p className="text-sm text-gray-500 mb-6">
        Average lifetime revenue per customer by how many campaigns they received before their first purchase.
      </p>

      <div className="bg-white rounded-lg p-5 shadow-sm border border-gray-100 mb-6">
        <h2 className="text-sm font-semibold text-gray-500 mb-4">Avg Revenue by Attribution Segment</h2>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => [`$${v.toFixed(2)}`, 'Avg Revenue']} />
            <Bar dataKey="revenue" fill="#C85510" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wider text-gray-400">
            <tr>
              {['Segment','Customers','Avg Campaigns','Avg Revenue','Avg AOV','Open Rate','Click Rate'].map(h => (
                <th key={h} className="px-4 py-3 text-left">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {data.map(d => (
              <tr key={d.attribution_segment} className="hover:bg-gray-50">
                <td className="px-4 py-3 font-medium">{SEGMENT_LABELS[d.attribution_segment] || d.attribution_segment}</td>
                <td className="px-4 py-3">{d.customer_count}</td>
                <td className="px-4 py-3">{d.avg_campaigns_before_purchase}</td>
                <td className="px-4 py-3">${Number(d.avg_lifetime_revenue || 0).toFixed(2)}</td>
                <td className="px-4 py-3">${Number(d.avg_order_value || 0).toFixed(2)}</td>
                <td className="px-4 py-3">{((d.avg_open_rate || 0) * 100).toFixed(1)}%</td>
                <td className="px-4 py-3">{((d.avg_click_rate || 0) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
