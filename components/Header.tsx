import siteMetadata from '@/data/siteMetadata'
import Link from './Link'
import MobileNav from './MobileNav'
import ThemeSwitch from './ThemeSwitch'
import SearchButton from './SearchButton'
import HeaderNav from './journal/HeaderNav'
import { Rss } from './journal/icons'

const Header = () => {
  return (
    <header className="ds-header sticky top-0 z-20 border-b border-line">
      <div className="ds-container flex h-[68px] items-center gap-5">
        {/* 워드마크 */}
        <Link href="/" aria-label={siteMetadata.headerTitle} className="flex flex-shrink-0 items-baseline gap-2.5">
          <span className="whitespace-nowrap font-mono text-[21px] font-semibold tracking-[-0.03em] text-ink">
            /steps
            <span
              aria-hidden
              className="ml-[0.12em] inline-block h-[0.95em] w-[0.55em] bg-accent-text [vertical-align:-0.12em]"
            />
          </span>
        </Link>

        <div className="ml-auto flex items-center gap-2">
          <HeaderNav />
          <div className="hidden items-center gap-1 sm:flex">
            <SearchButton />
            <a
              href="/feed.xml"
              aria-label="RSS"
              title="RSS"
              className="inline-flex h-[38px] w-[38px] items-center justify-center rounded-md border-[1.5px] border-transparent text-ink transition-colors hover:bg-surface-2"
            >
              <Rss />
            </a>
          </div>
          <ThemeSwitch />
          <MobileNav />
        </div>
      </div>
    </header>
  )
}

export default Header
