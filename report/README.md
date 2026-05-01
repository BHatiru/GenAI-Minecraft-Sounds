# Final Report — IEEE Format

LaTeX sources for the GenAI-B final project report.

## Files
- `main.tex` — IEEEtran conference paper (~6 pages + refs)
- `refs.bib` — bibliography
- Figures pulled from `../presentation/*.png` via `\graphicspath`

## Build (Windows / TeX Live or MiKTeX)
```powershell
cd report
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```
Output: `main.pdf`

## Build (online)
Upload the `report/` folder + the referenced PNGs from `presentation/`
to Overleaf. The `\graphicspath{{../presentation/}}` line assumes the
folder layout is preserved.
