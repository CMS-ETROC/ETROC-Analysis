"""Convert a percent-format .py source into a .ipynb (no jupytext dependency).

Cell markers:  `# %%` opens a code cell, `# %% [markdown]` opens a markdown cell
(whose body is written as ordinary `#` comment lines).

This is authoring scaffolding for the two notebooks in this folder; it is not part of the
analysis and nothing in the notebooks imports it.
"""
import sys
from pathlib import Path

import nbformat as nbf


def build(src: Path, dst: Path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            md = "\n".join(ln[2:] if ln.startswith("# ") else ln[1:] if ln == "#" else ln
                           for ln in body.split("\n"))
            cells.append(nbf.v4.new_markdown_cell(md))
        else:
            cells.append(nbf.v4.new_code_cell(body))

    for line in src.read_text().split("\n"):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
    nb.metadata["language_info"] = dict(name="python")
    nbf.write(nb, str(dst))
    print(f"{dst.name}: {len(cells)} cells")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
