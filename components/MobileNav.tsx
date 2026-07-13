'use client'

import { useState } from 'react'
import Link from './Link'
import headerNavLinks from '@/data/headerNavLinks'
import { Menu, Close } from './journal/icons'

const MobileNav = () => {
  const [navShow, setNavShow] = useState(false)

  const onToggleNav = () => {
    setNavShow((status) => {
      if (status) {
        document.body.style.overflow = 'auto'
      } else {
        // Prevent scrolling
        document.body.style.overflow = 'hidden'
      }
      return !status
    })
  }

  return (
    <>
      <button
        aria-label="메뉴 열기"
        onClick={onToggleNav}
        className="inline-flex h-[38px] w-[38px] items-center justify-center rounded-md border-[1.5px] border-transparent text-ink transition-colors hover:bg-surface-2 sm:hidden"
      >
        <Menu size={22} />
      </button>
      <div
        className={`fixed left-0 top-0 z-30 h-full w-full transform bg-page duration-300 ease-in-out ${
          navShow ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="flex justify-end px-6 pt-6">
          <button
            className="inline-flex h-[38px] w-[38px] items-center justify-center rounded-md border-[1.5px] border-line text-ink"
            aria-label="메뉴 닫기"
            onClick={onToggleNav}
          >
            <Close size={22} />
          </button>
        </div>
        <nav className="mt-6">
          {headerNavLinks.map((link) => (
            <div key={link.title} className="px-10 py-4">
              <Link
                href={link.href}
                className="font-display text-3xl font-extrabold tracking-tight text-ink"
                onClick={onToggleNav}
              >
                {link.title}
              </Link>
            </div>
          ))}
        </nav>
      </div>
    </>
  )
}

export default MobileNav
