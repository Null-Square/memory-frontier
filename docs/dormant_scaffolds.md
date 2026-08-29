# Dormant scaffolds for delayed predictive structure

The counterfactual-state result has a stronger higher-order consequence. A hard
memory state can be unreachable in every forward trajectory yet still form part
of a **dormant chain** through which a straight-through derivative propagates.
If the chain depth matches the temporal location of a predictive dependency, the
chain restores a first-order learning signal that a fully collapsed controller
cannot see.

## Delay-matched construction

Use the binary delayed-repeat source of delay `R>=2` and switch probability
`rho`. Adjacent and all shorter-than-`R` token lags are independent, while

\[
P(X_{t+R}=X_t)=1-\rho.
\]

Its centered predictive mode has eigenvalue

\[
\lambda=1-2\rho.
\]

Give the learner `K=R+1` memory states. From reset state 0, keep the forward
controller fully collapsed:

\[
F(0,x)=0.
\]

Install the following unreachable chain:

\[
1\to2\to\cdots\to R\to0,
\]

independently of the observed symbols. Because state 1 is never entered in the
hard forward pass, every nonzero state remains behaviorally unreachable.

Let decoder rows `0,...,R-1` predict the uniform token marginal. Perturb only the
final dormant decoder with logits

\[
(+\varepsilon,-\varepsilon).
\]

A derivative-induced transition from state 0 into dormant state 1 then sees only
uniform decoders for the first `R-1` future predictions and reaches the
specialized decoder exactly at lag `R`.

## First-order pressure theorem

Use the repository's standard equal-margin hard transition embedding: the hard
target logit has margin `a` over every alternative, and the backward softmax
temperature is `tau`. Define

\[
J_K(a,\tau)
=
\frac{t(s+1-t)}{\tau},
\]

where

\[
s=\frac{e^{a/\tau}}{e^{a/\tau}+K-1},\qquad
 t=\frac{1}{e^{a/\tau}+K-1}.
\]

For horizon `T>R`, let

\[
p_x=g_{0,x,0}-g_{0,x,1}
\]

be the STE descent pressure toward entering the dormant chain after token `x`.
Then around `epsilon=0`,

\[
\boxed{
\frac{\partial p_0}{\partial\varepsilon}\bigg|_0
=
+J_{R+1}(a,\tau)
\frac12\frac{T-R}{T}(1-2\rho),
}
\]

and

\[
\boxed{
\frac{\partial p_1}{\partial\varepsilon}\bigg|_0
=
-J_{R+1}(a,\tau)
\frac12\frac{T-R}{T}(1-2\rho).
}
\]

The result includes the full `K`-way softmax Jacobian. At first order, direct
counterfactual routing into the later dormant states contributes no centered
signal because those routes expose only lags `<R`, where the delayed-repeat
source is independent. Thus the only first-order nonconstant target value is the
proper chain entrance.

For a persistent delayed source (`rho<1/2`), any infinitesimal nonzero final
contrast therefore creates the correct opposite routing pressures for the two
tokens. Flipping the sign of the contrast only swaps the eventual memory labels.

## Depth selectivity

If the dormant chain has depth `D<R`, the specialized decoder is encountered at
lag `D`, where the source has no predictive token information. Consequently

\[
\boxed{
\frac{\partial p}{\partial\varepsilon}\bigg|_0=0
\quad\text{for }D<R.
}
\]

The regression suite checks both statements for delays up to `R=5` against the
NumPy exact-gradient oracle.

This gives an exact form of **temporal scaffold matching**:

- full collapse: higher-order dependency is locally inaccessible;
- too-shallow dormant chain: still first-order blind;
- delay-matched dormant chain: first-order gradient access is restored.

All three parameterizations execute the same forward memory trajectory from
reset until gradient descent actually changes the reachable row.

## Interpretation

The forward model does not possess the delayed-memory computation yet. Instead,
initialization can contain a behaviorally invisible computational scaffold whose
counterfactual depth determines which temporal dependencies are visible to
backpropagation.

This sharpens the project's capacity/accessibility distinction. A useful
finite-state algorithm can exist in the capacity frontier while the training
parameterization exposes it only if the dormant counterfactual automaton has an
appropriate temporal structure.
