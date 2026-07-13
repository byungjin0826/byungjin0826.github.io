'use client'

import { usePathname } from 'next/navigation'
import Link from '@/components/Link'
import headerNavLinks from '@/data/headerNavLinks'

function isActive(pathname: string, href: string) {
  if (href === '/') return pathname === '/'
  return pathname === href || pathname.startsWith(href + '/')
}

export default function HeaderNav() {
  const pathname = usePathname() || '/'
  return (
    <nav className="hidden items-center gap-1 sm:flex">
      {headerNavLinks.map((link) => {
        const active = isActive(pathname, link.href)
        return (
          <Link
            key={link.title}
            href={link.href}
            className={`px-2.5 text-[0.9375rem] font-semibold transition-colors ${
              active ? 'text-ink' : 'text-ink-muted hover:text-ink'
            }`}
          >
            {link.title}
          </Link>
        )
      })}
    </nav>
  )
}
