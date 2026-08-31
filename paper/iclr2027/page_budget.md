# ICLR 2027 nine-page main-text budget

The initial ICLR submission must fit within nine main-text pages. The working target is **8.4–8.6 pages**, leaving roughly 0.4–0.6 page for template-dependent movement, caption growth, and final citation corrections.

The appendix is unlimited but optional for reviewers. Therefore every premise required to understand or believe the central claim must remain in the main text; only derivation detail and secondary robustness material may be exported.

## Proposed page allocation

| Main-text material | Target pages | Non-negotiable content |
|---|---:|---|
| Abstract + introduction | 1.15 | capacity/behavior/accessibility separation; central forward-equivalence thesis; novelty boundary; compact contributions |
| Finite-memory setup + construction orders | 1.20 | affine local family; finite-horizon polynomial; readout quotient; definitions of support/operator/loss order |
| Hierarchy theorem + cancellation/genericity | 1.15 | theorem statement, one-paragraph proof, coefficient factorization, path vs decoder cancellation, generic equality qualification; Figure 2 |
| Dormant forward equivalence + same-forward spectrum | 1.25 | source-aware active rows, induction theorem, deterministic scaffold mechanism, fixed-class census protocol/counts; Figures 1 and 3 as space permits |
| Order-to-dynamics map | 1.20 | beneficial sign condition, isolated affine monomial law, finite/log/power hierarchy, metric-dependent interpretation |
| Parameterization + softmax boundary | 1.10 | regular-chart order invariance, singular boundary caveat, rare-edge softmax `delta^{-d}` law and full-K bound; Figure 4 |
| Smooth linear-SSM validation | 0.75 | same zero predictor, exact `g=prod w`, loss/gradient order 1..5, exact-zero cutoff; only concise illustrative SGD numbers if space remains |
| Related work + scope boundary | 0.70 | FSC gradients/polynomial objectives, same-function geometry, homogeneous dynamics, singular/dead directions, softmax slowness; near-tie caveat |
| Limitations + conclusion | 0.45 | finite horizon, local construction origin, isolated-route time maps, no adaptive-optimizer theorem, no claim of universal route winners |
| **Total target** | **8.95 before compression** | Must compress to 8.4–8.6 in official style |

The table deliberately starts slightly over the desired safety target. Compression should come from prose and figure packing, not from deleting theorem assumptions.

## Figure packing plan

### Figure 1 — dormant scaffold mechanism

Use a **single-column or 0.75-column-width conceptual panel** early in the dormant-equivalence section. If three subpanels are too wide at ICLR font size, replace them with two representative orders and state the full 1..R family in text.

### Figure 2 — order hierarchy

Use a compact single-column schematic adjacent to the theorem. This figure can replace several explanatory sentences and should therefore save space overall.

### Figure 3 — fixed-class census

Prefer a half-width histogram/table hybrid. If the four-figure layout exceeds the budget, the exact count table can replace the plot in main text and the full figure can move to the appendix. The scientific evidence is the frozen census protocol/counts, not the bar-chart rendering.

### Figure 4 — structural order versus optimizer geometry

Keep in main text. The two-panel affine/softmax comparison is the fastest way to prevent an optimizer-invariance misreading. It has higher priority than Figure 3.

## Compression order

When the official-style page count is known, compress in this order:

1. remove manuscript meta-language and repeated thesis sentences;
2. collapse the six-item contribution list to four bullets;
3. convert repeated equations already visible in figures to inline references;
4. shorten related-work exposition while preserving attribution and claim boundaries;
5. move the decoder-cancellation numerical fixture details to Appendix B, retaining only `(1,1,2)` in main text;
6. move the exact census-generation parameters to Appendix C, retaining the fixed-class statement and counts;
7. move full binary-logit antiderivative to Appendix F, retaining velocity and asymptotic time law;
8. move illustrative SGD threshold counts to Appendix G if needed;
9. move Figure 3 to appendix only if the exact count table remains in the main text.

Do **not** compress by removing:

- finite-horizon qualification;
- construction-origin qualification;
- decoder quotient definition;
- visibility-versus-beneficial-descent distinction;
- metric/parameterization dependence of time;
- prior-art boundary around polynomial FSC objectives and known homogeneous dynamics.

## Intended main-text section order

1. Introduction
2. Finite-memory construction order
3. Forward-equivalent dormant topology
4. Construction order and optimization geometry
5. Smooth recurrent validation
6. Related work and limitations
7. Conclusion

This is intentionally flatter than the canonical Markdown manuscript. The ICLR version should read as one causal argument, not as a chronological catalogue of project results.
