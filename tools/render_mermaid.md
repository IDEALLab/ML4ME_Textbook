# Diagrams (Mermaid / Graphviz) in the book

Mermaid and Graphviz blocks render in the browser for the HTML book, but for the
PDF Quarto has to rasterise them with headless Chrome. That works on a laptop
with Chrome installed and **hangs on the GitHub Actions runner**, which is why
the PDF never uses a live diagram block.

Pattern used in the notebooks (see `part1/taking_derivatives.ipynb`):

````markdown
::: {.content-visible when-format="html"}
```{mermaid}
graph LR
    A --> B
```
:::

::: {.content-visible unless-format="html"}
![Caption for the print edition.](figures/my_diagram.png){width=80%}
:::
````

## Regenerating the PNG after editing a diagram

1. Put the diagram block alone in a scratch `graph.qmd` (outside the repo):

   ```markdown
   ---
   format:
     html:
       mermaid-format: png
   ---

   ```{mermaid}
   ... diagram source ...
   ```
   ```

2. Run `quarto render graph.qmd`. Quarto uses your local Chrome and writes
   `graph_files/figure-html/mermaid-figure-1.png` (about 2300 px wide).
3. Copy that file over the PNG referenced in the notebook (for the example above,
   `part1/figures/computational_graph.png`) and commit it.

`python tools/check_latex_compat.py` warns about any diagram block that is not
wrapped in a `when-format="html"` div.
