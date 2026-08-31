# Bibliography verification audit

This file records the submission-facing verification status of `paper/references.bib`. The goal is to prevent citation metadata drift and, more importantly, to make sure the manuscript attributes prior results at the correct publication level rather than citing convenient reposts or preprints when a peer-reviewed version exists.

## Verified entries

### `aberdeen2002scaling`

**Scaling Internal-State Policy-Gradient Methods for POMDPs** — Douglas Aberdeen and Jonathan Baxter.

Verified against the authors' publication record / Google Research record and the original paper metadata:

- ICML 2002 / 19th International Conference on Machine Learning;
- pages 3--10;
- Morgan Kaufmann;
- year 2002.

A 2025 arXiv repost exists, but the bibliography correctly cites the original 2002 conference paper.

### `boularias2009predictive`

**Predictive Representations for Policy Gradient in POMDPs** — Abdeslam Boularias and Brahim Chaib-draa.

Verified against institutional/version-of-record metadata:

- ICML 2009;
- pages 65--72;
- ACM;
- DOI `10.1145/1553374.1553383`.

The abstract explicitly states that the FSC/PSR value function is polynomial, so this is the correct prior-art citation for the manuscript's statement that polynomial controller objectives themselves are not novel.

### `braziunas2004stochastic`

**Stochastic Local Search for POMDP Controllers** — Darius Braziunas and Craig Boutilier.

Verified against AAAI/author publication records:

- Nineteenth National Conference on Artificial Intelligence (AAAI-04);
- pages 690--696;
- San Jose, 2004.

### `fukumizu2019semiflat`

**Semi-flat minima and saddle points by embedding neural networks to overparameterization** — Kenji Fukumizu, Shoichiro Yamaguchi, Yoh-ichi Mototake, Mirai Tanaka.

Verified against the NeurIPS 2019 proceedings:

- Advances in Neural Information Processing Systems 32;
- year 2019;
- arXiv identifier 1906.04868 is consistent with the proceedings paper.

### `saxe2014exact`

**Exact solutions to the nonlinear dynamics of learning in deep linear neural networks** — Andrew M. Saxe, James L. McClelland, Surya Ganguli.

Verified against author/archival records:

- ICLR 2014;
- arXiv 1312.6120.

This remains background for known homogeneous/deep-linear learning dynamics rather than a novelty claim of the present paper.

### `amari1998natural`

**Natural Gradient Works Efficiently in Learning** — Shun-ichi Amari.

Verified against MIT Press:

- *Neural Computation* 10(2), 251--276 (1998);
- DOI `10.1162/089976698300017746`.

### `kunin2020neural`

**Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics** — Daniel Kunin, Javier Sagastuy-Brena, Surya Ganguli, Daniel L. K. Yamins, Hidenori Tanaka.

The previous bibliography represented this only as an arXiv 2020 preprint. Conference/proceedings records show that it was published at **ICLR 2021**. The canonical BibTeX has therefore been corrected to an `@inproceedings` ICLR 2021 entry while retaining arXiv 2012.04728. The stable internal citation key is intentionally left unchanged to avoid unnecessary source churn.

### `tanaka2021noether`

**Noether's Learning Dynamics: Role of Symmetry Breaking in Neural Networks** — Hidenori Tanaka and Daniel Kunin.

Verified against NeurIPS proceedings / DBLP:

- NeurIPS 2021, volume 34;
- pages 25646--25660;
- arXiv 2105.02716.

### `kunin2025alternating`

**Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks** — Daniel Kunin et al.

Verified against the NeurIPS 2025 proceedings and proceedings metadata:

- Advances in Neural Information Processing Systems 38;
- pages 4951--4998;
- main conference track;
- arXiv 2506.06489.

### `shirodkar2026dead`

**Dead Directions: Geometric Singular Learning** — Tejas Pradeep Shirodkar.

Verified against arXiv 2606.05957 (June 2026). No peer-reviewed publication was found in this audit, so the arXiv form is appropriate.

### `li2023softmax`

**Softmax policy gradient methods can take exponential time to converge** — Gen Li, Yuting Wei, Yuejie Chi, Yuxin Chen.

Verified against Springer:

- *Mathematical Programming* 201, 707--802 (2023);
- DOI `10.1007/s10107-022-01920-6`;
- version of record published January 23, 2023.

The manuscript cites this as prior art that generic softmax policy-gradient dynamics can be extremely slow, not as a source for our finite-memory degree calculation.

### `rawal2026saddle`

**A Theory of Saddle Escape in Deep Nonlinear Networks** — Divit Rawal and Michael R. DeWeese.

Verified against arXiv 2605.01288 (May 2026). The paper explicitly derives an `r-2` small-initialization saddle-escape exponent in its stated regime, so the present manuscript must not claim that exponent as new general optimization mathematics.

## Remaining release check

Before the final PDF, run BibTeX under the official ICLR style and inspect the rendered reference list for:

- capitalization loss in mathematical/model acronyms;
- author-name rendering (especially hyphenated/accented names);
- missing conference/journal fields introduced by the venue `.bst`;
- duplicate preprint/conference forms;
- broken DOI/arXiv strings;
- uncited bibliography entries and cited-but-missing keys.

This audit verifies metadata and attribution, not every stylistic preference of the final ICLR bibliography style.
