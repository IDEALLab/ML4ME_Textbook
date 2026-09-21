# Diagrams (Mermaid / Graphviz) in the book

A live ```` ```{mermaid} ```` or ```` ```{dot} ```` block makes Quarto launch
headless Chrome whenever a non-HTML format (the PDF) is rendered, and it does so
**even when the block sits inside a `::: {.content-visible when-format="html"}`
div**, because diagrams are processed before conditional content is dropped.
On a laptop with Chrome this takes a second; on the GitHub Actions runner the
publish job hangs on that chapter with no error message.

So the book does not contain live diagram blocks. Each diagram is a committed
PNG, and its Mermaid source is kept next to it in a Markdown comment so it can
be regenerated:

````markdown
<!-- The figure below is rendered from this Mermaid source (recipe: tools/render_mermaid.md).
graph LR
    A --> B
-->
![Caption.](figures/my_diagram.png){width=80%}
````

`python tools/check_latex_compat.py` fails on any live diagram block outside a
comment.

## Regenerating a PNG after editing the source

1. Put the diagram alone in a scratch `graph.qmd` **outside the repo**:

   ````markdown
   ---
   format:
     html:
       mermaid-format: png
   ---

   ```{mermaid}
   graph LR
       A --> B
   ```
   ````

2. Run `quarto render graph.qmd`. Quarto uses your local Chrome and writes
   `graph_files/figure-html/mermaid-figure-1.png` (about 2300 px wide).
3. Copy it over the PNG the notebook references (for the computational graph in
   `part1/taking_derivatives.ipynb` that is `part1/figures/computational_graph.png`)
   and commit both the PNG and the updated comment.
