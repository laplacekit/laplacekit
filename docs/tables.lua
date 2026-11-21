-- Wraps every table in a box that scrolls sideways (`.table-scroll` in
-- assets/theme.scss), so a table wider than the screen scrolls by itself
-- instead of widening the page.

function Table(tbl)
  return pandoc.Div({ tbl }, { class = "table-scroll" })
end
