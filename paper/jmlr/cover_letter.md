# Draft cover letter — Journal of Machine Learning Research

Dear Editors,

Please consider our manuscript, **“Same Predictor, Different Learnability: Construction Order in Finite-State Memory,”** for publication in the *Journal of Machine Learning Research*.

The paper studies a separation between predictive capacity and gradient accessibility in learned memory systems. For finite-state predictors we derive an exact local construction-order framework. The main theorem decomposes the first useful local loss degree into a source-valid support cost, a quotient occupancy-operator order, and a scalar decoder-visible order. We further prove that transition wiring that is completely dormant under the current source-memory process can be changed without altering the current finite-horizon predictor, while changing the derivative order at which a future useful computation becomes visible after learning opens an entrance into that dormant region.

The paper combines exact finite-horizon algebra, adversarial cancellation examples, a randomized 1,000-controller audit within one fixed forward-equivalence class, parameterization-aware optimization analysis, and an independent differentiable linear state-space validation. The contribution is intended as an analytical framework for understanding when representable memory computations are locally accessible to gradient-based learning, rather than as a new neural architecture or a state-of-the-art benchmark result.

We have taken care to distinguish the contribution from prior work on finite-state-controller polynomial objectives and memory gradients, same-function neural embeddings, homogeneous/deep-linear dynamics, singular directions, and softmax policy-gradient slowness. The public repository contains exact regression tests, deterministic figure generation, frozen evidence programs, and a reviewer reproduction guide.

**Generative-AI assistance.** OpenAI ChatGPT materially assisted conceptual organization, mathematical derivation/proof presentation, adversarial test design, implementation/review, literature search, figure preparation, and manuscript drafting/editing. The manuscript contains an explicit disclosure. Mathematical claims were checked against derivations and/or exact regression tests, numerical claims were reproduced from frozen code paths, and literature claims were checked against primary sources where available. The authors take responsibility for the final content.

Before submission, please complete the following fields:

- Funding: [TO COMPLETE]
- Competing interests: [TO COMPLETE]
- Prior or concurrent versions requiring disclosure: [TO COMPLETE]
- Suggested Action Editors: [TO COMPLETE]
- Suggested reviewers without conflicts: [TO COMPLETE]

Thank you for considering the manuscript.

Sincerely,

[AUTHOR NAME(S)]
