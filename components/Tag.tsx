import { slug } from 'github-slugger'
import { Chip } from '@/components/journal/ui'

interface Props {
  text: string
}

const Tag = ({ text }: Props) => {
  return (
    <Chip href={`/tags/${slug(text)}`} variant="outline" className="mb-2 mr-2">
      #{text}
    </Chip>
  )
}

export default Tag
