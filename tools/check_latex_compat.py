"""Scan the book's chapters for math that MathJax accepts but the LaTeX/PDF build does not.

Usage (from the repo root):

    python tools/check_latex_compat.py            # all chapters listed in _quarto.yml
    python tools/check_latex_compat.py part2/gen_models/transformers.ipynb ...

Exit status is 1 when a fatal pattern is found, so it can be used as a pre-render check.

Patterns that are fatal for the PDF build:
- a blank line inside an amsmath environment (align, gather, ...) or inside $$ ... $$
- $ ... $ that pandoc does not recognise as math (space after the opening $, space
  before the closing $, digit right after the closing $, or a blank line inside),
  so raw \\frac etc. reach LaTeX outside math mode
- commands MathJax defines but LaTeX does not (\\R, \\gt, \\lt, \\bold, ...)
- \\text{...} containing an unescaped _ & % or #

Patterns that only degrade the PDF (reported as warnings):
- \\mathbf{<Greek>}: unicode-math has no glyph for it in the default math font; use \\boldsymbol
- raw HTML tags in Markdown cells (<img>, <iframe>, <video>): dropped from the PDF
- footnote labels ([^1]) reused across chapters: pandoc merges chapters for the PDF,
  so labels must be unique book-wide
- non-ASCII characters inside math (e.g. a literal x): use \\times etc.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GREEK = (
    "Phi|Sigma|Theta|Lambda|Gamma|Omega|Psi|Delta|Xi|Pi|alpha|beta|gamma|delta|epsilon|"
    "theta|lambda|mu|sigma|phi|psi|omega|tau|eta|xi|pi|rho|nu|zeta|kappa|varepsilon|varphi"
)
MATHJAX_ONLY = {
    r"\R": "use \\mathbb{R}",
    r"\N": "use \\mathbb{N}",
    r"\Z": "use \\mathbb{Z}",
    r"\gt": "use >",
    r"\lt": "use <",
    r"\bold": "use \\mathbf",
    r"\bm": "use \\boldsymbol",
    r"\argmin": "use \\operatorname*{arg\\,min}",
    r"\argmax": "use \\operatorname*{arg\\,max}",
    r"\mathscr": "needs the mathrsfs package",
    r"\cancel": "needs the cancel package",
    r"\DeclareMathOperator": "not allowed inside math in LaTeX",
    r"\intertext": "only valid in align, not aligned",
    r"\require": "MathJax only",
    r"\class": "MathJax only",
    r"\cssId": "MathJax only",
    r"\unicode": "MathJax only",
}
ENVS = "align|gather|multline|eqnarray|equation|flalign|alignat"


def chapters_from_config():
    """Return every chapter/appendix file listed in _quarto.yml, in order."""
    text = (ROOT / "_quarto.yml").read_text(encoding="utf-8")
    return re.findall(r"^\s*-\s*([\w./-]+\.(?:ipynb|qmd))\s*$", text, flags=re.M)


def markdown_cells(path):
    """Yield (cell_index, text) for the Markdown content of a chapter file."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "markdown":
                yield i, "".join(cell["source"])
    else:
        yield 0, path.read_text(encoding="utf-8")


def blank_out(pattern, text, flags=0):
    """Replace every match of pattern with spaces so positions are preserved."""
    return re.sub(pattern, lambda m: " " * len(m.group(0)), text, flags=flags)


def literal_dollars(text):
    """Positions of $ signs that pandoc's tex_math_dollars rule treats as literal text."""
    n = len(text)
    pos = 0
    while pos < n:
        i = text.find("$", pos)
        if i < 0:
            return
        if i > 0 and text[i - 1] == "\\":
            pos = i + 1
            continue
        j = text.find("$", i + 1)
        good = (
            j > i + 1
            and not text[i + 1].isspace()
            and not text[j - 1].isspace()
            and not (j + 1 < n and text[j + 1].isdigit())
            and not re.search(r"\n[ \t]*\n", text[i:j])
        )
        if good:
            pos = j + 1
        else:
            yield i
            pos = i + 1


def scan(path, findings, footnote_labels):
    rel = path.relative_to(ROOT).as_posix()
    for ci, text in markdown_cells(path):
        where = f"{rel} cell {ci}"
        # Fatal: blank line inside $$ ... $$
        for m in re.finditer(r"\$\$(.+?)\$\$", text, flags=re.S):
            if re.search(r"\n[ \t]*\n", m.group(1)):
                findings["FATAL blank line inside $$ ... $$"].append(where)
            # $$ becomes \[ ... \] in LaTeX, and a numbered environment cannot be nested in it
            env = re.search(rf"\\begin\{{({ENVS})\*?\}}", m.group(1))
            if env:
                findings[f"FATAL \\begin{{{env.group(1)}}} inside $$ ... $$: use aligned/gathered/split"].append(where)
        no_display = blank_out(r"\$\$(.+?)\$\$", text, re.S)
        no_code = blank_out(r"`[^`\n]*`", blank_out(r"```.*?```", no_display, re.S))
        # Fatal: bare environments with blank lines
        for m in re.finditer(rf"\\begin\{{({ENVS})(\*?)\}}(.*?)\\end\{{\1\2\}}", no_code, flags=re.S):
            if re.search(r"\n[ \t]*\n", m.group(3)):
                findings[f"FATAL blank line inside \\begin{{{m.group(1)}{m.group(2)}}}"].append(where)
        # Fatal: $ that pandoc will not parse as math
        for i in literal_dollars(no_code):
            ctx = no_code[max(0, i - 25): i + 40].replace("\n", "|")
            findings["FATAL $ not recognised as math by pandoc"].append(f"{where}: ...{ctx}...")
        # Commands and \text{} contents inside recognised math
        bodies = [m.group(1) for m in re.finditer(r"\$\$(.+?)\$\$", text, flags=re.S)]
        bodies += re.findall(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$", no_display)
        for body in bodies:
            for cmd in set(re.findall(r"\\[A-Za-z]+", body)):
                if cmd in MATHJAX_ONLY:
                    findings[f"FATAL {cmd} is not defined in LaTeX ({MATHJAX_ONLY[cmd]})"].append(where)
            for tm in re.finditer(r"\\(?:text|mbox|textbf|textit)\{([^{}]*)\}", body):
                if re.search(r"(?<!\\)[_&%#]", tm.group(1)):
                    findings["FATAL unescaped _ & % # inside \\text{}"].append(f"{where}: {tm.group(0)[:50]}")
            if re.search(rf"\\mathbf\{{\\({GREEK})\}}", body):
                findings["WARN \\mathbf{Greek}: use \\boldsymbol for the PDF math font"].append(where)
            non_ascii = sorted(set(ch for ch in body if ord(ch) > 127))
            if non_ascii:
                findings["WARN non-ASCII character inside math"].append(f"{where}: {''.join(non_ascii)}")
        # Raw HTML that the PDF drops
        for tag in ("img", "iframe", "video"):
            if re.search(rf"<{tag}\b", text, flags=re.I):
                findings[f"WARN raw HTML <{tag}> is dropped from the PDF"].append(where)
        # Footnote labels
        for lab in set(re.findall(r"\[\^([^\]]+)\]:", text)):
            footnote_labels[lab].append(rel)
    # Chapter titles: a YAML `title:` plus a level-1 heading gives two chapters (one empty);
    # more than one level-1 heading splits the notebook into several chapters.
    cells = list(markdown_cells(path))
    has_yaml_title = any(
        re.search(r"^---\s*\n.*?^title:.*?^---", t, flags=re.S | re.M) for _, t in cells[:2]
    )
    h1s = [
        l
        for _, t in cells
        for l in blank_out(r"```.*?```", t, re.S).splitlines()   # ignore comments in fenced code
        if re.match(r"^# (?!#)", l)
    ]
    if has_yaml_title and h1s:
        findings["WARN YAML title plus a level-1 heading: remove one (duplicate, empty chapter)"].append(f"{rel}: {h1s[0][:60]}")
    if len(h1s) > 1 and path.name != "index.qmd":   # the preface deliberately holds several unnumbered chapters
        findings["WARN more than one level-1 heading: each becomes its own chapter"].append(f"{rel}: {', '.join(h[:40] for h in h1s[1:])}")


def main(argv):
    files = argv or chapters_from_config()
    findings = defaultdict(list)
    footnote_labels = defaultdict(list)
    for f in files:
        path = ROOT / f
        if not path.exists():
            findings["WARN file listed in _quarto.yml does not exist"].append(f)
            continue
        scan(path, findings, footnote_labels)
    for lab, where in footnote_labels.items():
        if len(where) > 1:
            findings["FATAL footnote label reused across chapters"].append(f"[^{lab}]: {', '.join(where)}")
    fatal = False
    for kind in sorted(findings):
        fatal |= kind.startswith("FATAL")
        print(f"\n{kind} ({len(findings[kind])})")
        for item in findings[kind]:
            print(f"  - {item}")
    if not findings:
        print("No LaTeX-compatibility problems found.")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
