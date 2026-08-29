# Higher-order blindness at collapsed memory

The predictive-spectrum theorem has a stronger extension: at a fully collapsed hard memory state, the local straight-through gradient of a generic stationary unifilar source sees only the source's **one-token Markovization**, not its full predictive state.

This creates an exact separation between **memory capacity** and **local gradient accessibility**.

## One-token Markovization

For a stationary source define

\[
\mu_x=P(X_t=x)
\]

and

\[
Q_{xy}=P(X_{t+1}=y\mid X_t=x).
\]

For a higher-order stochastic language, `Q` is only an observable projection. Different histories ending in the same token may have very different future laws even though they are merged by `Q`.

Consider a hard `K`-state memory controller whose every transition initially targets memory state 0. Initialize decoder row 0 to the stationary token marginal `mu`. Let unused decoder row 1 have logits

\[
\log\mu+d.
\]

All other decoder rows equal row 0. Canonical transition logits give state 0 a common margin `a`; the backward softmax temperature is `tau`.

Write

\[
\kappa=\frac{T-1}{T}\frac{t(s+1-t)}{\tau},
\]

where

\[
s=\frac{e^{a/\tau}}{e^{a/\tau}+K-1},\qquad
 t=\frac{1}{e^{a/\tau}+K-1}.
\]

Then the exact descent pressure for routing observed token `x` from collapsed memory 0 into unused memory 1 is

\[
\boxed{
p=\kappa D_\mu\left(Qd-\psi_\mu(d)\mathbf1\right),
}
\]

with

\[
\psi_\mu(d)=\log\mathbb E_{Y\sim\mu}[e^{d_Y}].
\]

After dividing by stationary token mass and removing the `mu`-weighted constant mode,

\[
\boxed{
\Pi_\mu D_\mu^{-1}p
=\kappa Q\Pi_\mu d.
}
\]

The implementation in `higher_order.py` computes `mu` and `Q` exactly from a `UnifilarSource`, and regression tests match the formula on the original four-state aliasing witness.

## Why only one token of predictive structure appears

At full hard collapse every forward transition lands in memory 0. Under the canonical surrogate embedding, all current-memory transition rows are identical as well. A derivative-induced memory perturbation created after observing token `x` can therefore affect the **next** prediction, but the following memory update erases dependence on that perturbed current-memory distribution.

Consequently the local gradient can use

\[
P(X_{t+1}\mid X_t=x)
\]

but no deeper history-conditioned predictive information. This is an architectural/optimization property of the collapsed parameter point, not a statement that the source itself is first-order Markov.

## Delayed-repeat blindness family

For alphabet size `q`, delay `R>=2`, and noise `rho`, define a source state as the previous `R` tokens. The next token copies the **oldest** stored token with probability `1-rho`; otherwise it switches uniformly to another symbol. The state then shifts and appends the new token.

The stationary state distribution is uniform over the `q^R` histories. Therefore adjacent tokens are independent:

\[
\mu_y=\frac1q,
\qquad
Q_{xy}=\frac1q
\quad\text{for all }x,y,
\]

while the next token remains strongly predictable from the token `R` steps back.

A `q^R`-state shift-register controller stores the required history and reaches the Bayes entropy rate

\[
h_q(\rho)=-(1-\rho)\log(1-\rho)-\rho\log\frac{\rho}{q-1}.
\]

The collapsed predictor has loss

\[
L_{collapsed}=\log q.
\]

Hence the exact achievable memory gain is

\[
\Delta=\log q-h_q(\rho),
\]

which approaches `log q` as `rho -> 0`.

Yet because `Q=1 mu^T`, any decoder contrast can be shifted to satisfy `mu^T d=0`, after which

\[
Qd=0.
\]

Jensen's inequality gives

\[
\psi_\mu(d)\ge0,
\]

with equality only for a constant contrast, which represents the same decoder distribution as the collapsed row. Therefore every nontrivial unused decoder has

\[
\boxed{p_x<0\quad\text{for every token }x.}
\]

At identical decoder rows, `p=0`. So random decoder asymmetry cannot rescue the collapsed memory: any genuine decoder specialization is locally pushed **away** from being used.

This is stronger than the earlier symmetry trap. There, a tiny decoder perturbation generated a useful transition signal when one-step predictive correlation was present. Here the source has substantial higher-order memory value but no local first-split route for that value to enter the discrete memory.

## Interpretation: capacity versus accessibility

The family gives an exact witness of

\[
\text{useful finite-state algorithm exists}
\quad\not\Rightarrow\quad
\text{gradient has a local path toward using memory}.
\]

We call these two objects:

- **capacity frontier:** the best prediction achievable with a given finite-state memory budget;
- **accessibility frontier:** which memory distinctions are locally reachable from a specified training parameterization and initialization.

The delayed-repeat family has a large capacity gain but zero first-split accessibility from full collapse.

This should be positioned carefully relative to prior work. Long-term dependency and parity problems have long demonstrated that recurrent networks can be hard to optimize, and recent parity theory shows strong initialization dependence. The narrower claim here is the exact source-conditioned mechanism: the collapsed hard-memory STE field factors through the one-token Markovization, yielding a closed-form higher-order blindness witness with a known Bayes-optimal finite-state controller and a known avoidable loss gap.
