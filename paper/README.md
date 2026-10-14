# Manuscript sources

Title: **ECG-based Structural Heart Disease Screening via Fully Convolutional Masked Pretraining and a One-Dimensional ConvNeXt**.

The source files and figures are copied from the local manuscript delivery `bspc_overleaf_20260917`, including the English revisions and table-width edits made on 2026-10-08. They are existing research drafts, not an assertion of publication or acceptance.

| Entry | Language | Engine | Layout |
|---|---|---|---|
| `main_en_singlecolumn.tex` | English | pdfLaTeX | Single column |
| `main_en_twocolumn.tex` | English | pdfLaTeX | Two columns |
| `main_zh_singlecolumn.tex` | Chinese | XeLaTeX | Single column |
| `main_zh_twocolumn.tex` | Chinese | XeLaTeX | Two columns |

Upload this directory to Overleaf and select the desired entry file and engine. With a local TeX distribution, run from this directory:

```bash
pdflatex main_en_twocolumn.tex
bibtex main_en_twocolumn
pdflatex main_en_twocolumn.tex
pdflatex main_en_twocolumn.tex
```

For Chinese, replace `pdflatex` with `xelatex` and select the Chinese stem. The included template, BibTeX database, custom bibliography style and all referenced figure files travel together.

The latest table-width edits were not compiled again during repository packaging. This release checks referenced source assets and preserves their hashes; it does not claim a newly rendered PDF. Editable SVGs, vector PDFs and PNG previews are in `figures/`. Original template license headers remain in the bundled files.
