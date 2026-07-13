import Link from './Link'
import siteMetadata from '@/data/siteMetadata'
import { Github, Mail, Rss } from './journal/icons'

function FLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="block py-1 text-[0.8125rem] text-ink-muted transition-colors hover:text-ink"
    >
      {children}
    </Link>
  )
}

export default function Footer() {
  return (
    <footer className="mt-24 border-t-2 border-rule">
      <div className="ds-container flex flex-wrap justify-between gap-8 py-12">
        <div className="max-w-[340px]">
          <div className="font-mono text-[19px] font-semibold tracking-[-0.03em] text-ink">
            /steps
            <span
              aria-hidden
              className="ml-[0.12em] inline-block h-[0.95em] w-[0.55em] bg-accent-text [vertical-align:-0.12em]"
            />
          </div>
          <div className="mt-4 flex gap-3.5 text-ink-muted">
            <a href={siteMetadata.github} target="_blank" rel="noreferrer" aria-label="GitHub" className="transition-colors hover:text-ink">
              <Github size={20} />
            </a>
            <a href={`mailto:${siteMetadata.email}`} aria-label="이메일" className="transition-colors hover:text-ink">
              <Mail size={20} />
            </a>
            <a href="/feed.xml" aria-label="RSS" className="transition-colors hover:text-ink">
              <Rss size={20} />
            </a>
          </div>
        </div>

        <div className="flex gap-12">
          <div>
            <div className="ds-eyebrow mb-3">글</div>
            <FLink href="/blog">전체 글</FLink>
            <FLink href="/tags">태그</FLink>
            <FLink href="/">홈</FLink>
          </div>
          <div>
            <div className="ds-eyebrow mb-3">더보기</div>
            <FLink href="/blog">아카이브</FLink>
          </div>
        </div>
      </div>

      <div className="border-t border-line">
        <div className="ds-container flex flex-wrap justify-between gap-2 py-4 font-mono text-xs text-ink-subtle">
          <span>© {new Date().getFullYear()} /steps. All rights reserved.</span>
          <span>Built with /steps Design System</span>
        </div>
      </div>
    </footer>
  )
}
