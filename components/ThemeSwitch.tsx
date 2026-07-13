'use client'

import { useEffect, useState } from 'react'
import { useTheme } from 'next-themes'
import { Sun, Moon } from '@/components/journal/icons'

const ThemeSwitch = () => {
  const [mounted, setMounted] = useState(false)
  const { theme, setTheme, resolvedTheme } = useTheme()

  // When mounted on client, now we can show the UI
  useEffect(() => setMounted(true), [])

  const isDark = mounted && (theme === 'dark' || resolvedTheme === 'dark')

  return (
    <button
      aria-label="테마 전환"
      title="테마 전환"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className="inline-flex h-[38px] w-[38px] items-center justify-center rounded-md border-[1.5px] border-line text-ink transition-colors hover:border-line-strong hover:bg-surface-2"
    >
      {/* mounted 전에는 아이콘을 숨겨 하이드레이션 불일치 방지 */}
      {mounted ? isDark ? <Sun /> : <Moon /> : <span className="h-[18px] w-[18px]" />}
    </button>
  )
}

export default ThemeSwitch
