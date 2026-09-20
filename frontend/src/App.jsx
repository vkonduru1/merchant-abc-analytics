// App.jsx
import { NavLink, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Customers from './pages/Customers'
import Campaigns from './pages/Campaigns'
import Analytics  from './pages/Analytics'

const NAV = [
  { to: '/',           label: 'Dashboard'   },
  { to: '/customers',  label: 'Customers'   },
  { to: '/campaigns',  label: 'Campaigns'   },
  { to: '/analytics',  label: 'Attribution' },
]

export default function App() {
  return (
    <>
      {/* Google Fonts */}
      <link
        href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;1,6..72,400&display=swap"
        rel="stylesheet"
      />

      <div className="flex min-h-screen" style={{ background: '#FBF8F2', fontFamily: "'Space Grotesk', system-ui, sans-serif" }}>

        {/* Sidebar */}
        <div className="w-[220px] flex-shrink-0 flex flex-col py-6" style={{ background: '#17274C' }}>
          {/* Logo */}
          <div className="px-5 pb-7" style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            <div className="text-[18px] font-bold text-white tracking-tight">☕ merchant-abc</div>
            <div className="text-[10px] tracking-widest uppercase mt-1" style={{ color: 'rgba(255,255,255,0.4)' }}>
              Analytics Platform
            </div>
          </div>

          {/* Nav */}
          <nav className="px-3 py-4 flex flex-col gap-1">
            {NAV.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `block px-3 py-2.5 rounded-lg text-[13px] transition-colors ${
                    isActive
                      ? 'font-semibold text-white'
                      : 'text-white/60 hover:text-white/90 hover:bg-white/5'
                  }`
                }
                style={({ isActive }) => isActive ? { background: '#C85510' } : {}}
              >
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Footer */}
          <div className="mt-auto px-5 pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            <div className="text-[10px] tracking-wide uppercase" style={{ color: 'rgba(255,255,255,0.3)' }}>AIIR Framework</div>
            <div className="text-[10px] mt-1" style={{ color: 'rgba(255,255,255,0.2)' }}>Built with Claude</div>
          </div>
        </div>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-8">
          <Routes>
            <Route path="/"           element={<Dashboard />} />
            <Route path="/customers"  element={<Customers />} />
            <Route path="/campaigns"  element={<Campaigns />} />
            <Route path="/analytics"  element={<Analytics />} />
          </Routes>
        </main>
      </div>
    </>
  )
}
