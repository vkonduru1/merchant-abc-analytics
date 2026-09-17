// Customers.jsx
import { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Customers() {
  const [customers, setCustomers] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get(`${API}/customers/?limit=20`)
      .then(r => { setCustomers(r.data.customers); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <p className="text-gray-400 text-sm">Loading…</p>

  return (
    <div>
      <h1 className="text-2xl font-bold text-brand-navy mb-6">Customers</h1>
      <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wider text-gray-400">
            <tr>
              {['ID', 'Email', 'Name', 'Orders', 'Total Spent', 'Marketing'].map(h => (
                <th key={h} className="px-4 py-3 text-left">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {customers.map(c => (
              <tr key={c.customer_id} className="hover:bg-gray-50">
                <td className="px-4 py-3 font-mono text-xs text-gray-400">{c.customer_id}</td>
                <td className="px-4 py-3">{c.email}</td>
                <td className="px-4 py-3">{c.first_name} {c.last_name}</td>
                <td className="px-4 py-3">{c.orders_count}</td>
                <td className="px-4 py-3">${Number(c.total_spent_usd || 0).toFixed(2)}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded-full text-xs ${
                    c.accepts_marketing ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                  }`}>
                    {c.accepts_marketing ? 'Yes' : 'No'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
