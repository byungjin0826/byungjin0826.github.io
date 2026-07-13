import siteMetadata from '@/data/siteMetadata'
import { genPageMetadata } from 'app/seo'
import { Button, Chip, Avatar, Divider, Eyebrow } from '@/components/journal/ui'
import { Arrow, Github, Mail } from '@/components/journal/icons'

export const metadata = genPageMetadata({ title: 'About', description: '/steps — 데이터·AI 엔지니어' })

const WRITES: [string, string][] = [
  [
    '기술',
    '실무와 공부의 기록 — RNN부터 GPT까지의 딥러닝 구조, 이상탐지·시계열 방법론, React2Shell 보안 사고 대응, 서버·인프라 이슈, AI를 실제로 써본 경험까지.',
  ],
  ['일상', '보홀 스쿠버 다이빙, 홋카이도 겨울 여행처럼 직접 걷고 겪은 것들.'],
]

const NOW: [string, string][] = [
  ['업무', '사내 LLM 서비스와 RAG 인프라 구축. IaC로 배포 자동화.'],
  ['관심', '시계열 예측의 본질적 어려움, 파운데이션 모델의 양면성.'],
  ['취미', '스쿠버 다이빙과 여행. 다음 목적지를 고르는 중.'],
]

export default function AboutPage() {
  return (
    <div className="ds-prose-container py-16">
      <Eyebrow>About</Eyebrow>
      <h1 className="mt-3.5 text-[3rem] font-extrabold leading-[1.1] tracking-[-0.03em] text-ink">
        안녕하세요,
        <br />
        /steps입니다.
      </h1>
      <p className="mt-5 text-[1.1875rem] leading-relaxed text-ink-muted">
        데이터·AI 엔지니어. RAG·LLM 인프라와 시계열 예측, 그리고 개발하며 겪은 것들을 기록합니다. 데이터로
        세상을 읽고, 여행으로 세상을 걷고, 새 기술로 내일을 상상합니다.
      </p>

      <div className="mt-7 flex flex-wrap gap-3">
        <Button href="/#subscribe" variant="primary">
          구독하기 <Arrow size={16} />
        </Button>
        <Button href={siteMetadata.github} variant="secondary" target="_blank" rel="noreferrer">
          <Github size={16} /> GitHub
        </Button>
        <Button href={`mailto:${siteMetadata.email}`} variant="ghost">
          <Mail size={16} /> 연락
        </Button>
      </div>

      {/* 무엇을 씁니다 */}
      <div className="mt-12">
        <Divider label="무엇을 씁니다" />
        <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2">
          {WRITES.map(([t, d]) => (
            <div key={t} className="border-t-2 border-rule pt-4">
              <Chip variant="solid">{t}</Chip>
              <p className="mt-3 text-[0.9375rem] leading-relaxed text-ink-muted">{d}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 요즘 */}
      <div className="mt-12">
        <Divider label="요즘 (Now)" />
        <ul className="mt-5 flex list-none flex-col gap-3.5 p-0">
          {NOW.map(([k, v]) => (
            <li key={k} className="flex gap-5">
              <span className="w-16 flex-shrink-0 font-mono text-[0.8125rem] font-semibold text-accent-text">
                {k}
              </span>
              <span className="text-[0.9375rem] text-ink">{v}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* 연락 카드 */}
      <div className="mt-12 flex flex-wrap items-center gap-5 rounded-lg border border-line bg-surface p-6">
        <Avatar name="/steps" size="lg" />
        <div className="min-w-[200px] flex-1">
          <div className="text-[1.1875rem] font-extrabold text-ink">같이 이야기해요</div>
          <p className="mt-1 text-[0.8125rem] text-ink-muted">
            데이터·기술·여행, 무엇이든. 편하게 연락 주세요 — {siteMetadata.email}
          </p>
        </div>
        <Button href={`mailto:${siteMetadata.email}`} variant="primary">
          이메일 보내기
        </Button>
      </div>
    </div>
  )
}
