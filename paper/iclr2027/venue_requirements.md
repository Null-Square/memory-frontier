# ICLR 2027 submission requirements

This directory is the venue-specific build workspace for the provisional ICLR 2027 submission of **Same Predictor, Different Learnability: Construction Order in Finite-State Memory**.

## Deadlines

Official ICLR 2027 deadlines are expressed in Anywhere on Earth (AoE):

- **Abstract submission:** September 18, 2026, 11:59 PM AoE.
- **Full paper submission:** September 25, 2026, 11:59 PM AoE.

The abstract deadline is hard. The registered abstract must be genuine, all authors must already be present, and no new authors can be added after that deadline. Titles/abstracts and author order may still be edited until the full-paper deadline under the conference rules.

## OpenReview and reviewing gates

These are desk-reject risks and should be resolved before manuscript polishing becomes the critical path:

- Every author must have an up-to-date OpenReview profile.
- A new OpenReview profile created without an institutional email can require moderation taking **up to two weeks**, so profile creation/update should be treated as urgent relative to the September 18 abstract deadline.
- ICLR 2027 requires every submission to have at least one author registered to review at least three papers, subject to the venue's qualification rules based on accepted publications at listed venues.
- If none of the authors is qualified under that definition, the submission is exempt from that reviewer-qualification requirement, but the venue's one-paper cap for papers with no eligible reciprocal reviewer still applies.
- Authors appearing on three or more ICLR 2027 submissions have an additional reviewing obligation unless exempt as organizers.

The paper repository cannot determine author eligibility; this must be checked against the actual author list and current OpenReview profiles before abstract registration.

## Format and anonymity

- Submission is **double blind**.
- Main text is limited to **9 pages** at initial submission.
- References do not count toward the main-text limit.
- Appendices may follow the references and are not counted in the main-text limit, but reviewers are not required to read them.
- Use the **official ICLR 2027 style package** from:
  `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`
- Do not substitute an unofficial mirror for the final submission build.
- Keep the anonymous author block in submission mode; do not enable the final/camera-ready switch.
- Any author identity revealed in main text or supplementary material can trigger desk rejection.

The main scientific argument must therefore stand on its own inside nine pages. The appendix should contain proof details and robustness material, not premises required to understand the headline theorem.

## Required paper-facing statements

### AI-use disclosure

ICLR 2027 requires an explicit generative-AI-use disclosure section in the manuscript and corresponding submission-form disclosure. This project used generative AI in tasks that the policy identifies as requiring disclosure, including conceptual/theoretical development, mathematical claims and proof assistance, experiment design, implementation/review, interpretation, literature search, figure preparation, and manuscript drafting/editing.

The draft is in `ai_use_statement.tex`. It must remain truthful and should not be weakened for optics.

### Reproducibility statement

ICLR strongly encourages a reproducibility statement at the end of the main text before references. The draft in `reproducibility_statement.tex` points to exact tests, frozen census protocols, figure-generation regressions, and the claim-to-code matrix. The venue states that this statement does not count toward the main-text page limit.

## Submission source layout

The intended build is:

- `main.tex` — anonymous venue shell and section order;
- `sections/` — compressed ICLR main-text sections;
- `ai_use_statement.tex` — mandatory disclosure;
- `reproducibility_statement.tex` — reproducibility statement;
- `appendix.tex` — venue supplement assembled from the canonical appendix sources;
- `../references.bib` — canonical bibliography;
- `figures/` — generated vector figures copied from a clean release build;
- official `iclr2027_conference.sty` and bibliography style copied locally from the official ICLR archive for the final build only.

The venue build must not fork scientific content silently. Canonical source-of-truth files remain:

- `paper/manuscript.md`
- `paper/appendix.md`
- `paper/appendix_dynamics.md`
- `paper/references.bib`
- `paper/reproducibility_matrix.md`

## Scientific claim boundary that must survive compression

Do not remove or blur any of the following to gain space:

1. the **finite-horizon** scope;
2. the **construction-origin** assumption for the hierarchy;
3. the decoder-readout equality quotient;
4. the distinction between **visibility order** and **beneficial descent order**;
5. the distinction between structural order and optimizer/parameterization-specific time;
6. the fact that the affine `d-2` and softmax `d` bootstrap exponents are specialized dynamical maps, not universal optimizer-invariant laws;
7. the novelty boundary against prior work on FSC polynomial objectives, same-function landscape embeddings, homogeneous saddle dynamics, balancedness, and generic softmax slowness.

## Build gates

A submission candidate is not ready until all of the following hold:

- all author OpenReview profiles are complete and reciprocal-review eligibility is checked;
- genuine abstract registered before September 18 AoE with the complete author set;
- official style package installed from the ICLR source;
- anonymous build compiles from a clean directory;
- main text is at most 9 pages with at least a small safety margin;
- no undefined citations/references or missing figures;
- AI-use disclosure present;
- reproducibility statement present;
- anonymous-code bundle passes the checklist in `anonymous_code_checklist.md`;
- four main figures are generated by the regression-tested figure script;
- exact tests pass at the release SHA;
- the two frozen outside-CI evidence programs reproduce;
- every quantitative claim has an entry in `paper/reproducibility_matrix.md`.
