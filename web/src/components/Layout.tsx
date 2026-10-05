import { BarChart3, Cpu, Layers, LayoutGrid, Menu, Presentation, ScanSearch, ShieldCheck, Sigma, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router'
import { PROJECT } from '@/content'
import { MonoLabel } from './ui'

const NAV: { section: string; items: { to: string; label: string; icon: typeof Cpu }[] }[] = [
  { section: 'Project', items: [{ to: '/', label: 'Overview', icon: LayoutGrid }, { to: '/demo', label: 'Live demo', icon: ScanSearch }] },
  { section: 'Method', items: [{ to: '/design', label: 'Design', icon: Layers }, { to: '/quantization', label: 'Quantization', icon: Sigma }, { to: '/checker', label: 'Checker', icon: ShieldCheck }] },
  { section: 'Evidence', items: [{ to: '/results', label: 'Results', icon: BarChart3 }, { to: '/hardware', label: 'Hardware', icon: Cpu }] },
]

/** The mark: a chip outline in ink with the die in accent. */
export function Logo({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect x="6" y="6" width="20" height="20" rx="3" fill="var(--color-ink)" />
      <rect x="11" y="11" width="10" height="10" rx="1.5" fill="var(--color-accent)" />
      <g stroke="var(--color-ink)" strokeWidth="2" strokeLinecap="round">
        <path d="M11 2v3M16 2v3M21 2v3M11 27v3M16 27v3M21 27v3M2 11h3M2 16h3M2 21h3M27 11h3M27 16h3M27 21h3" />
      </g>
    </svg>
  )
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <NavLink to="/" onClick={onNavigate} className="flex items-center gap-2 px-[18px] pb-3.5 pt-4">
        <Logo className="h-[18px] w-[18px]" />
        <span className="text-body font-bold tracking-[-0.01em] text-ink">{PROJECT.short}</span>
      </NavLink>
      <nav className="flex-1 space-y-[18px] overflow-y-auto px-2 pt-2">
        {NAV.map((g) => (
          <div key={g.section}>
            <MonoLabel className="mb-1 px-2.5">{g.section}</MonoLabel>
            {g.items.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} end={to === '/'} onClick={onNavigate}
                className={({ isActive }) =>
                  `group flex h-[30px] items-center gap-2.5 rounded-md px-2.5 text-body transition-colors ${isActive ? 'bg-accent-soft font-medium text-ink' : 'text-ink-2 hover:bg-hover hover:text-ink'}`}>
                {({ isActive }) => (
                  <>
                    <Icon size={14} strokeWidth={1.75} className={isActive ? 'text-accent' : 'text-ink-3'} />
                    <span className="flex-1 truncate">{label}</span>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div className="space-y-3 border-t border-line px-[18px] py-4">
        <NavLink to="/present" className="btn btn-secondary w-full"><Presentation size={14} /> Present</NavLink>
        <p className="text-[12px] leading-snug text-ink-3">
          Accelerator: ztachip by Vuong Nguyen (Apache-2.0). Models, tooling and this site are ours.
        </p>
      </div>
    </div>
  )
}

export default function Layout() {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)
  useEffect(() => { window.scrollTo(0, 0); setOpen(false) }, [pathname])
  return (
    <div className="min-h-screen bg-canvas">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[230px] border-r border-line bg-canvas lg:block">
        <Sidebar />
      </aside>
      <div className="sticky top-0 z-30 flex h-12 items-center gap-2 border-b border-line bg-canvas/90 px-4 backdrop-blur lg:hidden">
        <button className="btn btn-icon" onClick={() => setOpen(true)} aria-label="Open navigation"><Menu size={16} /></button>
        <Logo className="h-4 w-4" /><span className="text-body font-bold">{PROJECT.short}</span>
      </div>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/10" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-[260px] border-r border-line bg-canvas">
            <button className="btn btn-icon absolute right-2 top-3" onClick={() => setOpen(false)} aria-label="Close navigation"><X size={15} /></button>
            <Sidebar onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
      <main className="lg:pl-[230px]">
        <div className="mx-auto max-w-[1080px] px-7 pb-24">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
