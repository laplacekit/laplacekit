-- Sets the name like the wordmark: "laplace" in the text colour, "kit" bold in
-- the brand colour, both in the wordmark face (assets/brand.scss).
-- Prose writes **laplacekit**; code spans and plain text are left alone.

local function wordmark()
  return pandoc.Span({ pandoc.Str("laplace"), pandoc.Strong({ pandoc.Str("kit") }) }, { class = "brand" })
end

function Strong(el)
  if #el.content == 1 and el.content[1].t == "Str" and el.content[1].text == "laplacekit" then
    return wordmark()
  end
end
