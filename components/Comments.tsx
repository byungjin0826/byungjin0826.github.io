'use client'

import { Comments as CommentsComponent } from 'pliny/comments/index.js'
import { useState } from 'react'
import siteMetadata from '@/data/siteMetadata'
import { Button } from '@/components/journal/ui'

export default function Comments({ slug }: { slug: string }) {
  const [loadComments, setLoadComments] = useState(false)

  // giscus env 미설정이면 렌더하지 않음 (빈 위젯 방지)
  if (siteMetadata.comments?.provider === 'giscus' && !siteMetadata.comments.giscusConfig?.repo) {
    return null
  }

  return (
    <>
      {!loadComments && (
        <Button variant="secondary" size="sm" onClick={() => setLoadComments(true)}>
          댓글 불러오기
        </Button>
      )}
      {siteMetadata.comments && loadComments && (
        <CommentsComponent commentsConfig={siteMetadata.comments} slug={slug} />
      )}
    </>
  )
}
