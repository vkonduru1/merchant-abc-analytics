import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard.jsx'
import Customers from './pages/Customers.jsx'
import Analytics from './pages/Analytics.jsx'
import Campaigns from './pages/Campaigns.jsx'

function NavLink({ to, children }) {
  const location = useLocation()
  const active = location.pathname === to
  return (
    <Link
      to={to}
      className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
        active
          ? 'bg-brand-orange text-white'
          : 'text-white/70 hover:text-white hover:bg-white/10'
      }`}
    >
      {children}
    </Link>
  )
}

export default function App() {
  return (
    <div className="min-h-screen bg-brand-paper">
      {/* Nav */}
      <nav className="bg-brand-navy px-6 py-4 flex items-center gap-6">
        <span className="text-white font-bold text-lg tracking-tight mr-4">
          ☕ merchant-abc
        </span>
        <NavLink to="/">Dashboard</NavLink>
        <NavLink to="/customers">Customers</NavLink>
        <NavLink to="/campaigns">Campaigns</NavLink>
        <NavLink to="/analytics">Attribution</NavLink>
      </nav>

      {/* Content */}
      <main className="p-6">
        <Routes>
          <Route path="/"           element={<Dashboard />} />
          <Route path="/customers"  element={<Customers />} />
          <Route path="/campaigns"  element={<Campaigns />} />
          <Route path="/analytics"  element={<Analytics />} />
        </Routes>
      </main>
    </div>
  )
}
