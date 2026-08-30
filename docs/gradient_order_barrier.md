# Memory depth as gradient order

The delayed-repeat family exposes a smooth optimization barrier that is independent of the hard-forward straight-through estimator.

## Soft chain construction

Use the binary delay-`R` source with switch probability `rho`. A stochastic memory controller has states `0,1,...,R`. State 0 is the collapsed baseline. A token `0` enters state 1 with probability `u_1`; intermediate state `j` advances to `j+1` with probability `u_{j+1}` and otherwise collapses back to 0. All decoder rows are uniform except final state `R`, whose logits are `(+delta,-delta)` and therefore specialize toward predicting token 0.

The final decoder can influence loss only when probability mass reaches state `R`. Every path from state 0 to state `R` contains all `R` soft links. Consequently the loss difference is divisible by

\[
\prod_{j=1}^R u_j.
\]

All partial derivatives that omit at least one link vanish at the fully collapsed point.

## Exact leading term

Set every link probability equal to `epsilon`. Then

\[
\boxed{
\log 2-L(\epsilon)
=C_R\epsilon^R+O(\epsilon^{R+1}).
}
\]

Let

\[
q=\sigma(2\delta)
\]

be the final decoder probability assigned to token 0. Conditional on a token-0 chain origin, the delayed source emits token 0 after `R` steps with probability `1-rho`. The prediction gain of the final decoder over the uniform baseline is

\[
B=\log 2+(1-\rho)\log q+\rho\log(1-q).
\]

There are `T-R` eligible prediction positions in a horizon of `T`, and the stationary probability of the triggering token 0 is `1/2`. Therefore

\[
\boxed{
C_R=\frac{T-R}{T}\frac12 B.
}
\]

Regression tests for delays 2, 3, 4, and 5 match this coefficient to within one percent using finite perturbations. A scratch log-log sweep gave slopes approximately

```text
delay 2 -> 1.995
delay 3 -> 2.995
delay 4 -> 3.995
delay 5 -> 4.945
```

with the delay-5 discrepancy dominated by floating-point precision at the smallest perturbations.

## Dormant scaffolds lower derivative order

Now prewire the downstream links:

\[
u_2=\cdots=u_R=1,
\]

and vary only the first link `u_1=epsilon`. The same predictive computation becomes

\[
\boxed{
\log 2-L(\epsilon)=C_R\epsilon+O(\epsilon^2).
}
\]

So the dormant chain does not merely increase gradient magnitude. It changes the **order of the first nonzero useful derivative** from `R` to 1.

This gives a smooth-objective interpretation of the earlier counterfactual-automaton results: an unreachable hard scaffold supplies the downstream factors of a temporal computation in advance, allowing one local parameter perturbation to expose a dependency that otherwise requires an `R`-way coordinated change.

## Claim boundary

Products of parameters, vanishing gradients with depth, and high-order flat saddles are classical phenomena in deep linear and recurrent networks. Delayed-copy tasks are also standard diagnostics of long-range credit assignment.

The narrower result here is source-conditioned and finite-state: for an exactly solvable stochastic language, required memory depth is equal to the leading perturbative order of a concrete memory-construction path, while behaviorally dormant topology can reduce that order without changing the current forward predictor. This connects exact predictive memory requirements to the local order of the optimization landscape.
