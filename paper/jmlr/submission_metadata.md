# JMLR submission metadata

## Title

Same Predictor, Different Learnability: Construction Order in Finite-State Memory

## Running title

Construction Order in Finite-State Memory

## Keywords

1. finite-state memory
2. gradient accessibility
3. optimization geometry
4. recurrent learning
5. state-space models

## Manuscript category

Theoretical / analytical machine learning.

## Abstract

A memory architecture can contain a useful predictive computation while local optimization is poorly positioned to construct it. We make this separation exact for finite-state memory predictors. Around an affine controller transition family, finite-horizon prediction loss is an exact multivariate polynomial in local transition strengths. We define a source-valid support construction order, a quotient occupancy-operator order, and the first nonconstant scalar-loss order, proving d_support <= d_operator <= d_loss. Each scalar coefficient factors into a source/path construction operator paired with decoder log-probabilities, separating source/path cancellation from decoder cancellation. We also prove that rewiring controller rows unused by the current source-memory process leaves the entire current finite-horizon predictor unchanged, yet can change construction order once learning makes those rows reachable. One forward-equivalence class realizes orders one through five solely through dormant topology. Construction order fixes local homogeneity, while parameterization and metric determine its conversion to optimization time. A differentiable linear state-space delay model trained with autograd reproduces loss orders one through five from identical current predictors. Thus forward-equivalent memory systems can expose the same future computation to learning at radically different local orders.

## Reproducibility

Public repository: https://github.com/Null-Square/memory-frontier

The paper is accompanied by exact regression tests, frozen evidence programs, deterministic figure generation, a reviewer-oriented reproduction guide, and a CI-built reviewer code bundle.
