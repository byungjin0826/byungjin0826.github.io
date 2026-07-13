import { AlgoliaButton } from 'pliny/search/AlgoliaButton.js'
import { KBarButton } from 'pliny/search/KBarButton.js'
import siteMetadata from '@/data/siteMetadata'
import { Search } from '@/components/journal/icons'

const SearchButton = () => {
  if (
    siteMetadata.search &&
    (siteMetadata.search.provider === 'algolia' || siteMetadata.search.provider === 'kbar')
  ) {
    const SearchButtonWrapper =
      siteMetadata.search.provider === 'algolia' ? AlgoliaButton : KBarButton

    return (
      <SearchButtonWrapper
        aria-label="검색"
        title="검색"
        className="inline-flex h-[38px] w-[38px] items-center justify-center rounded-md border-[1.5px] border-transparent text-ink transition-colors hover:bg-surface-2"
      >
        <Search />
      </SearchButtonWrapper>
    )
  }
}

export default SearchButton
