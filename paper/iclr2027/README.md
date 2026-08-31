# ICLR 2027 build workspace

This directory contains the provisional anonymous ICLR 2027 port of the canonical construction-order manuscript.

## Current state

The first compression pass is complete:

- anonymous `main.tex` shell;
- seven main-text section files;
- main theorem/forward-equivalence proof appendix;
- dynamics/parameterization/SSM appendix;
- mandatory AI-use disclosure draft;
- reproducibility statement draft;
- nine-page budget;
- anonymous-code checklist;
- venue requirements/deadlines.

The scientific source of truth remains the canonical files one directory above. Changes to the venue text should be checked against those sources rather than allowed to drift independently.

## Official style requirement

The final submission must use the official ICLR 2027 archive. Download the conference-provided style package and place the required `.sty`/`.bst` files in this directory. Do not commit an unofficial mirrored style file as a substitute.

The source intentionally falls back to ordinary `article` mode when `iclr2027_conference.sty` is absent. **Fallback mode is only a syntax check. Its pagination is meaningless for the ICLR limit.**

## Syntax-check build

From this directory, after the section files exist:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Generated figures are expected under `figures/`. Until they are copied in, `main.tex` renders labeled placeholders rather than failing, which allows text syntax checking independently of figure generation.

## Final ICLR build

After installing the official style:

1. confirm the source is still in anonymous/submission mode;
2. generate the four figures from the frozen release commit;
3. copy the PDF figures into `figures/`;
4. compile with BibTeX using canonical `../references.bib`;
5. inspect the official-style main-text page count against `page_budget.md`;
6. inspect undefined references/citations, overfull boxes, figure readability, and font embedding;
7. verify AI-use and reproducibility statements are present in the policy-compliant location;
8. verify appendices follow the bibliography and no main theorem depends on appendix-only definitions;
9. prepare the anonymized code artifact using `anonymous_code_checklist.md`.

## Compression principle

The current port is intentionally shorter and flatter than `../manuscript.md`. Space should be recovered from repeated exposition, detailed fixture parameters, and derivations that are already in the appendix. Do not recover space by deleting theorem scope conditions or by turning metric-specific optimization laws into universal claims.
