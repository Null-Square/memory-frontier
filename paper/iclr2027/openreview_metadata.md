# ICLR 2027 OpenReview metadata draft

This file is the copy-verified source for abstract registration. It contains no
author identities. Confirm the live OpenReview form fields/options before final
submission rather than inferring dropdown labels from this document.

## Title

**Same Predictor, Different Learnability: Construction Order in Finite-State Memory**

## Genuine abstract

A memory architecture can contain a useful predictive computation while local
optimization is poorly positioned to construct it. We make this separation exact
for finite-state memory predictors. Around an affine controller transition
family, finite-horizon prediction loss is an exact multivariate polynomial in
local transition strengths. We define a source-valid support construction order,
a quotient occupancy-operator order, and the first nonconstant scalar-loss order,
and prove

\[
d_{\mathrm{support}}\le d_{\mathrm{operator}}\le d_{\mathrm{loss}}.
\]

Every scalar coefficient factors as
\(c_\alpha=-\langle G_\alpha,\log q\rangle\), separating source/path cancellation
from decoder cancellation. We then prove that changing controller rows that the
current source-memory process cannot exercise leaves the complete finite-horizon
forward process unchanged, yet such dormant rewiring can change construction
order after a learned entrance makes those rows reachable. One exact
forward-equivalence class realizes orders one through five when only dormant
topology varies. Construction order determines local homogeneity, whereas its
conversion to optimization time depends on parameterization and metric: isolated
affine-probability flow has the known finite/logarithmic/
\(\delta^{-(d-2)}\) hierarchy, while rare-edge Euclidean softmax flow scales as
\(\delta^{-d}\). An independent linear state-space delay model trained with
ordinary autograd reproduces loss orders one through five from identical current
zero predictors. Thus forward-equivalent memory systems can expose the same
future useful computation to learning at radically different local orders.

## Candidate keywords

These are candidate free-text keywords; confirm the live form vocabulary:

- finite-state memory
- gradient accessibility
- non-convex optimization
- recurrent models
- state-space models
- local optimization geometry
- representation / computation construction

## Candidate subject-area emphasis

Based on the published ICLR 2027 call, the strongest topical emphasis is:

1. non-convex optimization / optimization for learning;
2. general machine learning;
3. structured or compositional learning, where the form offers a compatible label.

Use the live OpenReview taxonomy as authoritative.

## Abstract-registration gates

Before the September 18, 2026 11:59 PM AoE abstract deadline:

- [ ] title matches the current anonymous PDF source;
- [ ] abstract is copied from this file or from the identical `main.tex` abstract;
- [ ] all intended authors are present (new authors cannot be added after the abstract deadline);
- [ ] every intended author has a complete OpenReview profile linked to the submission email;
- [ ] reciprocal-review eligibility/registration obligations have been checked against the actual author list;
- [ ] no placeholder or duplicate abstract is submitted.

Before the September 25, 2026 11:59 PM AoE full-paper deadline:

- [ ] author order is final;
- [ ] title/abstract edits, if any, are synchronized back to the repository;
- [ ] anonymous PDF and anonymous code artifacts come from the same green submission-build SHA;
- [ ] AI-use form disclosure matches `ai_use_statement.tex`;
- [ ] supplementary code ZIP is the workflow-produced anonymized namespace build;
- [ ] final PDF passes page-count, citation/reference, figure, metadata, and identity scans.
