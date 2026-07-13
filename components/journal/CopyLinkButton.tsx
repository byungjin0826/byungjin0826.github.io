'use client'

import { useState } from 'react'
import { Link as LinkIcon } from '@/components/journal/icons'

export default function CopyLinkButton() {
  const [copied, setCopied] = useState(false)
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard 불가 환경 무시 */
    }
  }
  return (
    <button
      type="button"
      onClick={onCopy}
      className="inline-flex items-center justify-center gap-1.5 rounded-md border-[1.5px] border-transparent px-3.5 py-[7px] text-[0.8125rem] font-semibold text-ink transition-colors hover:bg-surface-2"
    >
      <LinkIcon size={15} /> {copied ? '복사됨!' : '링크 복사'}
    </button>
  )
}
