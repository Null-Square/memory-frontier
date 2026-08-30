# Submission-gap audit after manuscript v2

This audit is intentionally adversarial. The goal is to decide whether the
research still needs new science or whether the remaining work is manuscript
engineering.

## Executive assessment

Current state:

- **core finite-memory theorem:** closed;
- **forward-equivalence intervention:** closed;
- **genericity/cancellation boundary:** closed;
- **same-forward randomized breadth:** closed;
- **independent smooth recurrent validation:** closed;
- **regular-coordinate objection:** closed;
- **simplex-boundary/softmax objection:** closed;
- **leading-route near-degeneracy objection:** closed;
- **related-work collision audit:** no direct collision found, but claim wording
  must remain narrow;
- **visibility versus beneficial descent semantics:** now closed by explicit sign
  condition.

The remaining high-value work is mostly exposition, figures, citation completion,
and reproducibility packaging. New theory should be added only if a concrete
reviewer objection cannot be answered from existing results.

## Gate matrix

| gate | status | severity if unresolved | action |
|---|---|---:|---|
| construction-order hierarchy proof | PASS | critical | no new theory |
| dormant forward-equivalence proof | PASS | critical | no new theory |
| fixed-direction same-forward census | PASS | high | promote to main figure |
| strict cancellation witness | PASS | high | keep near theorem |
| beneficial-sign/descent condition | PASS | high | wording + short proposition |
| regular coordinate invariance | PASS | high | concise main-text robustness |
| simplex-boundary softmax bridge | PASS | high | revise dynamics section |
| independent smooth recurrent model | PASS | high | keep as validation, not novelty |
| prior FSC polynomial-degree overlap | PASS/KNOWN | high | state global-vs-local distinction |
| same-function embedding overlap | PASS/KNOWN | high | do not claim generic same-function novelty |
| dormant-feature/singular-order overlap | PASS/KNOWN | high | keep finite-memory path/operator distinction |
| route/spectral near-tie scope | PASS | medium | one caution example, appendix details |
| infinite-horizon extension | OPEN/OPTIONAL | medium | limitation unless reviewer demands |
| large neural model experiment | OPEN/OPTIONAL | medium | not required for theory paper |
| adaptive optimizer theory | OPEN/OPTIONAL | low-medium | limitation |
| paper figures | OPEN | high for presentation | build next |
| final bibliography verification | OPEN | high for submission | build next |
| appendix assembly | OPEN | high | build after figures |
| theorem numbering/cross references | OPEN | medium | manuscript engineering |
| artifact/reproduction instructions | OPEN | medium-high | add final reproducibility table |

## Reviewer attack 1: “This is just graph distance.”

**Answer:** No. The structural path cost is only the lower layer
\(d_{\rm support}\). Exact source-weighted path cancellation can raise the quotient
operator order, and decoder cancellation can raise the scalar order again:

\[
 d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}.
\]

The explicit \((1,1,2)\) fixture demonstrates why graph distance alone is
insufficient.

**Status:** closed.

## Reviewer attack 2: “The two models are not really the same.”

**Answer:** The dormant forward-equivalence theorem proves equality of the entire
source-memory occupancy process throughout the horizon, not merely equality of one
loss value. The random census fixes source, architecture, decoder, reachable
dynamics, horizon, and trainable direction family; only dormant zero-cost wiring
changes.

**Status:** closed.

## Reviewer attack 3: “You changed the parameterization.”

**Answer:** The main dormant comparison changes the base topology, not the local
coordinate chart. Separately, the scalar loss-germ order is invariant under any
smooth local reparameterization with nonsingular Jacobian. Singular charts can
change order by weighted degree, and simplex boundaries require a separate metric
analysis.

**Status:** closed.

## Reviewer attack 4: “Your escape exponent is an artifact of probability coordinates.”

**Answer:** Correct that the absolute exponent is coordinate/metric dependent; the
paper now says so explicitly. The structural object is construction order/local
homogeneity. Direct probability flow maps degree \(d\) to the known finite/log/
\(d-2\) hierarchy, while rare-edge softmax flow maps it to
\(\Theta(\delta^{-d})\). Both retain a severe one-order scaffold advantage.

**Status:** closed by softmax-boundary result.

## Reviewer attack 5: “First nonzero does not mean useful.”

**Answer:** Correct in general. Define the admissible local perturbation cone
\(K\) and degree-leading homogeneous forms \(P_k\). The visibility order is
\(d_{\rm vis}=d_{\rm loss}\). The beneficial ray order is the smallest ray-leading
degree with negative coefficient. Then

\[
 d_\downarrow\ge d_{\rm vis},
\]

with equality iff the leading degree-\(d_{\rm vis}\) form is negative on some
admissible ray. The main construction-time witnesses explicitly satisfy this sign
condition.

**Status:** closed; integrate terminology into final manuscript.

## Reviewer attack 6: “FSC objectives were already known to be polynomial.”

**Answer:** Agreed. Prior FSC/PSR work studies polynomial objectives and how
representation structure affects global/maximal degree. The new object here is the
**minimum local nonconstant degree around a specified forward-equivalent base**,
with source-valid support, quotient occupancy, and decoder layers.

**Status:** overlap acknowledged; novelty claim remains narrower.

## Reviewer attack 7: “Same function, different landscape is old.”

**Answer:** Agreed. Same-function inactive-unit/inactive-propagation embeddings,
dormant-feature activation, and singular high-order directions are prior art. The
paper should never use the broad phrase as its contribution. The finite-memory
claim fixes the forward source-conditioned process and identifies a combinatorial
source-valid construction order plus an exact occupancy-operator factorization.

**Status:** overlap acknowledged.

## Reviewer attack 8: “Your result is one hand-picked chain.”

**Answer:** The deterministic chain gives arbitrary-order existence. The stronger
1,000-instance census holds the trainable directions and all forward-active
objects fixed and randomizes only dormant wiring, producing orders one through
five with zero hierarchy violations. A separate smooth linear state-space memory
reproduces orders one through five under PyTorch autograd.

**Status:** closed.

## Reviewer attack 9: “Leading order cannot predict what is learned.”

**Answer:** Not without a margin. Higher-order terms break leading balance laws and
can reverse nearly tied scalar routes or nearly degenerate singular modes. The
error scale is controlled by the gap between leading construction degree and the
first symmetry/route-breaking degree. The paper uses one exact reversal to state
the scope boundary rather than claiming global trajectory prediction.

**Status:** closed.

## Reviewer attack 10: “Finite horizon is too artificial.”

The finite horizon is what makes the complete objective an exact finite
polynomial and permits coefficient-level proofs. Extending the theorem to an
infinite-horizon stationary objective would require interchange of the perturbative
series with the stationary/mixing limit and control of the stationary distribution
as transition parameters vary.

Potential value of an infinite-horizon extension: real but not necessary for the
current theorem paper. It should be listed as future work unless a target venue or
reviewer specifically demands it.

**Status:** optional. Do not expand the project now.

## Reviewer attack 11: “Why not show a large RNN/SSM?”

The linear-state-space experiment already answers the narrow external-validity
question: the multiplication/path-order mechanism survives outside the finite
controller formalism under ordinary autograd. A large model would add ecological
validity but would weaken causal isolation and substantially expand experimental
scope.

**Status:** optional. Add only if targeting an empirically demanding venue.

## What must happen before submission

### Required

1. Integrate manuscript-v2 wording into the canonical paper file.
2. Build four main figures:
   - same-forward dormant scaffold diagram;
   - support/operator/loss hierarchy and cancellation diagram;
   - fixed-class order census;
   - parameterization-dependent order-to-time curves.
3. Verify every bibliography entry against a primary source.
4. Assemble theorem proofs into a numbered appendix and cross-reference them.
5. Add a reproducibility table mapping every main numerical statement to a test or
   experiment script.
6. Run one final clean checkout/test/experiment smoke pass from the release commit.

### Recommended

1. Add the linear-SSM scaling plot as an appendix figure.
2. Include exactly one near-tie route reversal in the main paper or supplement.
3. Add a compact table distinguishing capacity, forward behavior, visibility
   order, beneficial descent order, and optimizer-time law.
4. Have an independent reader attempt to falsify the theorem assumptions from the
   manuscript alone.

### Not currently justified

- more balance-law theorems;
- more route-race examples;
- a general adaptive-optimizer theory;
- a general multiway-softmax exact prefactor theorem;
- a large neural benchmark suite;
- an infinite-horizon extension before the current paper is written.

## Current readiness estimate

Scientific core: **91–93%**.

Submission-ready manuscript/reproducibility package: **72–76%**.

The remaining gap is mostly paper construction, not uncertainty about whether the
central phenomenon exists. The main scientific risk is now novelty positioning,
not theorem correctness. The targeted audit has found adjacent and partially
overlapping ideas but no direct match for the complete source-valid
support/operator/dormant-rewire result. That should be stated as a careful search
outcome, not as proof of novelty.
