import Link from '@/components/Link'
import { Button } from '@/components/journal/ui'

export default function NotFound() {
  return (
    <div className="ds-container flex flex-col items-start justify-start py-16 md:mt-24 md:flex-row md:items-center md:justify-center md:space-x-6">
      <div className="space-x-2 pb-8 pt-6 md:space-y-5">
        <h1 className="font-display text-6xl font-extrabold leading-9 tracking-tight text-ink md:border-r-2 md:border-rule md:px-6 md:text-8xl md:leading-14">
          404
        </h1>
      </div>
      <div className="max-w-md">
        <p className="mb-4 text-xl font-bold leading-normal text-ink md:text-2xl">
          페이지를 찾을 수 없습니다.
        </p>
        <p className="mb-8 text-ink-muted">
          하지만 걱정 마세요 — 홈에서 다른 글들을 둘러볼 수 있습니다.
        </p>
        <Button href="/" variant="primary">
          홈으로 돌아가기
        </Button>
      </div>
    </div>
  )
}
