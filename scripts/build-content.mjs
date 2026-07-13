// 빌드/dev 전에 실행 — contentlayer 대체 콘텐츠 생성기.
// fs 읽기는 이 순수 Node 스크립트에서만 수행하고, 앱은 생성된 JSON을 정적 import 한다.
// 생성물:
//   lib/generated/blog.json, lib/generated/authors.json  (앱 소비용)
//   app/tag-data.json, public/search.json, public/feed.xml, public/tags/<tag>/feed.xml
import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import readingTime from 'reading-time'
import GithubSlugger, { slug as ghSlug } from 'github-slugger'
import siteMetadata from '../data/siteMetadata.js'

const ROOT = process.cwd()
const DATA_DIR = path.join(ROOT, 'data')

function walk(dir) {
  if (!fs.existsSync(dir)) return []
  let out = []
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) out = out.concat(walk(p))
    else if (/\.mdx?$/.test(e.name)) out.push(p)
  }
  return out
}

function flattenedPath(file) {
  return path.relative(DATA_DIR, file).replace(/\.mdx?$/, '').split(path.sep).join('/')
}

function extractToc(raw) {
  const slugger = new GithubSlugger()
  const toc = []
  const re = /^(#{1,3})\s+(.+)$/gm
  let m
  while ((m = re.exec(raw))) {
    const value = m[2].replace(/[#*`]/g, '').trim()
    toc.push({ value, url: '#' + slugger.slug(value), depth: m[1].length })
  }
  return toc
}

function buildBlog(file) {
  const { data, content } = matter(fs.readFileSync(file, 'utf8'))
  const fp = flattenedPath(file)
  return {
    title: data.title,
    date: data.date ? new Date(data.date).toISOString() : new Date(0).toISOString(),
    tags: data.tags || [],
    lastmod: data.lastmod ? new Date(data.lastmod).toISOString() : undefined,
    draft: data.draft ?? false,
    summary: data.summary || '',
    images: data.images,
    authors: data.authors,
    layout: data.layout,
    bibliography: data.bibliography,
    canonicalUrl: data.canonicalUrl,
    slug: fp.replace(/^.+?\//, ''),
    path: fp,
    filePath: path.relative(DATA_DIR, file).split(path.sep).join('/'),
    readingTime: readingTime(content),
    toc: extractToc(content),
    structuredData: {
      '@context': 'https://schema.org',
      '@type': 'BlogPosting',
      headline: data.title,
      datePublished: data.date,
      dateModified: data.lastmod || data.date,
      description: data.summary,
      image: data.images
        ? Array.isArray(data.images)
          ? data.images[0]
          : data.images
        : siteMetadata.socialBanner,
      url: `${siteMetadata.siteUrl}/${fp}`,
    },
    body: { raw: content },
  }
}

function buildAuthor(file) {
  const { data, content } = matter(fs.readFileSync(file, 'utf8'))
  const fp = flattenedPath(file)
  return {
    name: data.name,
    avatar: data.avatar,
    occupation: data.occupation,
    company: data.company,
    email: data.email,
    twitter: data.twitter,
    linkedin: data.linkedin,
    github: data.github,
    layout: data.layout,
    slug: fp.replace(/^.+?\//, ''),
    path: fp,
    body: { raw: content },
  }
}

const escape = (s = '') =>
  String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

function rssItem(post) {
  return `
  <item>
    <guid>${siteMetadata.siteUrl}/blog/${post.slug}</guid>
    <title>${escape(post.title)}</title>
    <link>${siteMetadata.siteUrl}/blog/${post.slug}</link>
    ${post.summary ? `<description>${escape(post.summary)}</description>` : ''}
    <pubDate>${new Date(post.date).toUTCString()}</pubDate>
    <author>${siteMetadata.email} (${siteMetadata.author})</author>
    ${(post.tags || []).map((t) => `<category>${escape(t)}</category>`).join('')}
  </item>`
}

function rssFeed(posts, page = 'feed.xml') {
  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${escape(siteMetadata.title)}</title>
    <link>${siteMetadata.siteUrl}/blog</link>
    <description>${escape(siteMetadata.description)}</description>
    <language>${siteMetadata.language}</language>
    <managingEditor>${siteMetadata.email} (${siteMetadata.author})</managingEditor>
    <webMaster>${siteMetadata.email} (${siteMetadata.author})</webMaster>
    <lastBuildDate>${new Date(posts[0].date).toUTCString()}</lastBuildDate>
    <atom:link href="${siteMetadata.siteUrl}/${page}" rel="self" type="application/rss+xml"/>
    ${posts.map(rssItem).join('')}
  </channel>
</rss>`
}

function writeJson(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, JSON.stringify(data))
}

function main() {
  const allBlogs = walk(path.join(DATA_DIR, 'blog'))
    .map(buildBlog)
    .sort((a, b) => new Date(b.date) - new Date(a.date))
  const allAuthors = walk(path.join(DATA_DIR, 'authors')).map(buildAuthor)

  // 앱 소비용(초안 포함 — 앱이 프로덕션에서 필터)
  writeJson(path.join(ROOT, 'lib', 'generated', 'blog.json'), allBlogs)
  writeJson(path.join(ROOT, 'lib', 'generated', 'authors.json'), allAuthors)

  // 발행글(초안 제외)로 태그/검색/RSS
  const posts = allBlogs.filter((p) => p.draft !== true)

  const tagCount = {}
  for (const p of posts) for (const t of p.tags) tagCount[ghSlug(t)] = (tagCount[ghSlug(t)] || 0) + 1
  writeJson(path.join(ROOT, 'app', 'tag-data.json'), tagCount)

  const searchDocs = posts.map((p) => ({
    title: p.title,
    date: p.date,
    tags: p.tags,
    summary: p.summary,
    slug: p.slug,
    path: p.path,
  }))
  writeJson(path.join(ROOT, 'public', 'search.json'), searchDocs)

  if (posts.length > 0) {
    fs.writeFileSync(path.join(ROOT, 'public', 'feed.xml'), rssFeed(posts))
    for (const tag of Object.keys(tagCount)) {
      const filtered = posts.filter((p) => p.tags.map((t) => ghSlug(t)).includes(tag))
      if (filtered.length === 0) continue
      const dir = path.join(ROOT, 'public', 'tags', tag)
      fs.mkdirSync(dir, { recursive: true })
      fs.writeFileSync(path.join(dir, 'feed.xml'), rssFeed(filtered, `tags/${tag}/feed.xml`))
    }
  }

  console.log(
    `content: ${allBlogs.length} posts (${posts.length} published) · ${Object.keys(tagCount).length} tags · JSON + search + RSS generated`
  )
}

main()
