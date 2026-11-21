-- Points each page at its Markdown copy (`website: llms-txt`), so an agent that
-- fetches the HTML finds `<page>.llms.md` next to it.

function Meta(meta)
  local output = PANDOC_STATE.output_file
  if output == nil or not output:match("%.html$") then
    return nil
  end

  local page = output:match("([^/\\]+)%.html$")
  local link = string.format(
    '<link rel="alternate" type="text/markdown" href="%s.llms.md">', page
  )

  local includes = meta["header-includes"]
  if includes == nil then
    includes = pandoc.MetaList({})
  elseif includes.t ~= "MetaList" then
    includes = pandoc.MetaList({ includes })
  end

  includes:insert(pandoc.MetaBlocks({ pandoc.RawBlock("html", link) }))
  meta["header-includes"] = includes

  return meta
end
