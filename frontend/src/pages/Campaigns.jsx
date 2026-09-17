// Campaigns.jsx
import { useEffect, useState } from 'react'
import axios from 'axios'
import { format } from 'date-fns'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Campaigns() {
  const [campaigns, setCampaigns] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get(`${API}/campaigns/?limit=30`)
      .then(r => { setCampaigns(r.data.campaigns); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <p className="text-gray-400 text-sm">Loading…</p>

  return (
    <div>
      <h1 className="text-2xl font-bold text-brand-navy mb-6">Campaigns</h1>
      <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wider text-gray-400">
            <tr>
              {['Campaign','Type','Send Time','Recipients','Opens','Clicks','Discount'].map(h => (
                <th key={h} className="px-4 py-3 text-left">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {campaigns.map(c => (
              <tr key={c.campaign_id} className="hover:bg-gray-50">
                <td className="px-4 py-3 font-medium max-w-xs truncate">{c.campaign_name}</td>
                <td className="px-4 py-3">
                  <span className="px-2 py-0.5 rounded-full text-xs bg-blue-50 text-blue-600">
                    {c.campaign_type_name || '—'}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-500 text-xs">
                  {c.send_time ? format(new Date(c.send_time), 'MMM d, yyyy') : '—'}
                </td>
                <td className="px-4 py-3">{c.total_recipients?.toLocaleString()}</td>
                <td className="px-4 py-3">{c.total_opens?.toLocaleString()}</td>
                <td className="px-4 py-3">{c.total_clicks?.toLocaleString()}</td>
                <td className="px-4 py-3">
                  {c.has_discount_code
                    ? <span className="text-brand-orange font-medium">{c.discount_pct}% off</span>
                    : <span className="text-gray-300">—</span>
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
