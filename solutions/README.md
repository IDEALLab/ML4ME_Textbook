# Exercise solutions

One executed notebook per chapter, named `<chapter_slug>_solutions.ipynb`, where the
slug is the chapter notebook's file name (for example
`reviewing_supervised_linear_models_solutions.ipynb`). Each notebook is embedded
into `appendices/exercise_solutions.qmd`, which is always published with the book.

Conventions:

- One `## Exercise N` section per exercise, in the chapter's numbering.
- Reuse the chapter's own code and data; keep new code minimal.
- Notebooks are executed locally before committing (the site build does not run
  Python), and they contain no assertions or verification scaffolding.
