# Symmetry trap family

This note records an exact optimization pathology for the hard-forward categorical-memory learner. It is a theorem fixture, not a claim that unused discrete states or codebook collapse are new phenomena; VQ/VQ-VAE work has long documented dead-code under-utilization and revival heuristics. The point here is that the finite-state source gives exact predictive ground truth and an analytic avoidable-loss gap.

## Symmetric repeat source

For alphabet size `q >= 2` and total switch probability `rho in (0,1)`, let the source state be the previous symbol. From state `s`,

\[
P(X_{t+1}=s\mid S_t=s)=1-\rho,
\]

and every other symbol has probability `rho/(q-1)`. After symbol `x` is emitted,

\[
S_{t+1}=x.
\]

The stationary source-state and token marginals are uniform. The Bayes entropy rate is

\[
h_q(\rho)=-(1-\rho)\log(1-\rho)-\rho\log\frac{\rho}{q-1}.
\]

A `q`-state controller with `F(m,x)=x` stores the previous symbol and therefore achieves `h_q(rho)` asymptotically. No predictor can beat the Bayes source-state loss, so this is the exact asymptotic `K=q` optimum.

## Exact hard-forward stationary point

Now initialize a `q`-state hard controller so every `(memory, symbol)` transition targets memory state 0. Initialize every decoder row identically to the stationary token marginal, which is uniform for this source. With zero decoder logits this is the usual symmetric initialization.

Under the hard-forward straight-through training rule used in this repository:

1. Forward memory remains in state 0, so all other decoder rows are unoccupied.
2. The occupied decoder already predicts the unconditional token marginal, hence its expected cross-entropy gradient is zero.
3. Unoccupied decoder rows receive zero forward data gradient.
4. Since every decoder row is identical, changing the surrogate memory distribution cannot change any future prediction loss, hence the transition-logit gradient is also zero.

Therefore the joint parameter point is exactly stationary for every finite horizon and every positive backward softmax temperature.

Yet its loss is

\[
L_{collapsed}=\log q,
\]

while the exact asymptotic `K=q` optimum is

\[
L^*_{q}=h_q(\rho).
\]

The avoidable stationary-point gap is

\[
\Delta_q(\rho)=\log q-h_q(\rho),
\]

and

\[
\lim_{\rho\to0}\Delta_q(\rho)=\log q.
\]

Thus for fixed `q=2` the trap can hide almost one bit/token of useful memory, and across growing alphabets the avoidable gap grows as `log q`.

## Exact binary symmetry-breaking response

For `q=2`, let the transition-logit row for the currently selected target have margin `a` over the unused target, and let the backward softmax temperature be `tau`. Write

\[
s=\sigma(a/\tau).
\]

Perturb the two decoder rows antisymmetrically so their token-0-vs-token-1 log-odds are `-delta` and `+delta`. For the collapsed controller, define the straight-through descent pressure for changing the active transition after symbol `x` from memory 0 to memory 1 as

\[
P_x=g_{0,x,0}-g_{0,x,1}.
\]

Then the exact expected-gradient calculation gives

\[
P_0=\frac{T-1}{T}\frac{s(1-s)}{\tau}(1-2\rho)\,\delta,
\]

and

\[
P_1=-P_0.
\]

This is exact for the antisymmetric perturbation, not merely a first-order approximation. The identity follows because the difference in conditional cross-entropies between decoder log-odds `-delta` and `+delta` is exactly `(1-2 rho) delta`.

For a persistent source (`rho < 1/2`), any nonzero decoder contrast therefore pushes the two observed symbols toward opposite memory assignments. Reversing the perturbation swaps the memory labels but preserves the specialization. The predictive correlation `(1-2 rho)` is literally the gain on this symmetry-breaking signal.

At `rho=1/2`, the source has no predictive memory value and the pressure vanishes, as it should.

## Finite-horizon correction

With reset memory state 0, the last-symbol controller shares memory label 0 between the initial unknown state and genuine symbol 0. Its single decoder row therefore mixes those regimes at finite `T`. `symmetric_repeat_last_symbol_finite_horizon_loss` implements the exact shared-readout correction and is regression-tested against the general finite-horizon scorer.

The transient vanishes as `T -> infinity`, recovering `h_q(rho)`.

## Interpretation

The pathology is stronger than ordinary capacity failure: an optimal memory algorithm exists, the achievable improvement is known analytically, but first-order gradient training cannot leave the symmetric collapsed solution without an external symmetry-breaking signal. Decoder asymmetry, random initialization, code revival, entropy regularization, or other exploration mechanisms should therefore be treated as computational resources in optimizer comparisons rather than hidden implementation details.

The closest adjacent literature is vector-quantization/codebook collapse, where unused codes receive sparse or zero useful updates and are often revived or reparameterized. Our narrower object is a recurrent predictive-memory system with an exactly solvable source, exact finite-state optimum, and exact gradient response, so the claim should be framed as a controlled theorem/benchmark rather than as discovery of dead discrete codes in general.
