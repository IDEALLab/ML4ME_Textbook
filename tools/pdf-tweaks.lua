--[[
pdf-tweaks.lua: make the PDF edition of the book compact and readable.

Applied only when rendering to LaTeX/PDF (see the `filters:` entry of the pdf
format in _quarto.yml). Two things happen:

1. Code is hidden, except for cells that carry `#| code-fold: false`. Those cells
   were deliberately written to be read alongside the text, so they stay. Set
   `pdf-hide-code: false` in the document metadata to keep all code (the
   solutions book does this).

2. Outputs that only make sense in a browser (ipywidgets, Plotly, other HTML-only
   MIME types) are replaced by a one-line note instead of their raw text/plain
   fallback such as `interactive(children=(FloatSlider(...`.
]]

if not FORMAT:match("latex") then
  return {}
end

local hide_code = true

local function is_browser_only_output(text)
  return text:match("^interactive%(children=") ~= nil
    or text:match("^HBox%(children=") ~= nil
    or text:match("^VBox%(children=") ~= nil
    or text:match("^Output%(") ~= nil
    or text:match("^%u%w*%(value=") ~= nil            -- e.g. FloatSlider(value=...)
    or text:match("^Unable to display output for mime type") ~= nil
end

-- Outputs that carry no information in print, e.g. the return value of interact():
-- `<function __main__.plot_p_norm(p=2.0)>`, or a bare object repr.
local function is_noise_output(text)
  return text:match("^<function ") ~= nil
    or text:match("^<IPython%.") ~= nil
    or text:match("^<ipywidgets%.") ~= nil
    or text:match("^<matplotlib%.") ~= nil
end

local note = pandoc.Para({
  pandoc.Emph({ pandoc.Str("[Interactive output: available in the online version of this chapter.]") }),
})

return {
  {
    Meta = function(meta)
      if meta["pdf-hide-code"] ~= nil then
        hide_code = pandoc.utils.stringify(meta["pdf-hide-code"]) ~= "false"
      end
    end,
  },
  {
    CodeBlock = function(el)
      if el.classes:includes("cell-code") then
        if hide_code and el.attributes["code-fold"] ~= "false" then
          return {}
        end
        return nil
      end
      if is_browser_only_output(el.text) then
        return note
      end
      if is_noise_output(el.text) then
        return {}
      end
      return nil
    end,
  },
}
