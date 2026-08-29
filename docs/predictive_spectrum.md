# Predictive spectrum and collapsed-memory gradient flow

This note records an exact operator identity for the hard-forward straight-through memory learner. The claim is deliberately narrow: transfer-operator / Markov spectral methods are classical, and straight-through estimators are classical. The new object studied here is the exact map from an observable source's predictive dynamics to the surrogate gradient that breaks a collapsed discrete memory state.

## Setup

Let `P` be a `q x q` doubly stochastic Markov matrix. The observable source state is the previous emitted symbol, so

\[
P(X_{t+1}=y\mid X_t=x)=P_{xy},
\]

and after emitting `y`, the next source state is exactly `y`. Double stochasticity makes the stationary token law uniform.

Use a `q`-state hard memory controller whose every transition initially targets memory state 0. Canonical transition logits assign target 0 margin `a>0` over every alternative, while the backward softmax temperature is `tau>0`.

All decoder rows are initially identical except row 1. Let

\[
d=r_1-r_0\in\mathbb{R}^q
\]

be the decoder-logit contrast between memory rows 1 and 0. Rows `2,...,q-1` equal row 0.

For each observed token `x`, define the straight-through descent pressure toward routing that token from memory 0 to memory 1 as

\[
p_x=g_{0,x,0}-g_{0,x,1},
\]

where `g` is the exact expected finite-horizon gradient with respect to transition logits.

Let

\[
\Pi=I-\frac{1}{q}\mathbf 1\mathbf 1^\top
\]

remove the uniform token mode.

## Exact pressure operator

Write

\[
s=\frac{e^{a/\tau}}{e^{a/\tau}+q-1},
\qquad
 t=\frac{1}{e^{a/\tau}+q-1}.
\]

For a reset sequence with `T` predictions define

\[
C=\frac{T-1}{T}\frac{1}{q}\frac{t(s+1-t)}{\tau}>0.
\]

Then

\[
\boxed{\Pi p=C\,P\,\Pi d.}
\]

The identity is exact for finite decoder-logit contrast `d`; it is not only a linearization around `d=0`.

### Proof sketch

At the collapsed hard controller, every forward transition lands in memory 0. Under the canonical embedding, every soft transition row is also identical. Therefore a perturbation of the surrogate memory distribution created by one transition is erased by the next transition; only the immediately following prediction contributes to the transition gradient.

For a token `x`, the difference in next-step cross-entropy between decoder rows 0 and 1 is

\[
\mathrm{CE}(P_x,q_0)-\mathrm{CE}(P_x,q_1)
= P_x d + c,
\]

where `c` is the decoder log-normalizer difference and is independent of `x`. The softmax Jacobian contributes the common positive factor `t(s+1-t)/tau`, the stationary frequency of each token contributes `1/q`, and only `T-1` of `T` prediction positions have a preceding memory transition. Centering over `x` removes `c`. Since `P` is doubly stochastic, centering commutes with `P`, giving the stated identity.

## Spectral consequence

If `d` is a centered right eigenvector of `P`,

\[
Pd=\lambda d,
\]

then

\[
\boxed{\Pi p=C\lambda d.}
\]

Thus the source's predictive eigenvalue directly sets both the magnitude and sign of the specialization pressure:

- large positive `lambda`: decoder contrast is amplified into same-direction memory specialization;
- `lambda=0`: that decoder mode produces no centered transition pressure;
- negative `lambda`: the specialization pressure reverses direction.

For a general non-normal doubly stochastic `P`, the operator statement remains valid while singular values, rather than eigenvalue magnitudes alone, control norm amplification. For symmetric/reversible sources the eigenbasis gives the cleanest interpretation.

## Four-mode regression fixture

The test suite includes a positive `4 x 4` symmetric Markov matrix with orthonormal Walsh modes and eigenvalues

\[
1,\quad 0.5,\quad 0.2,\quad -0.1.
\]

For finite decoder contrasts, `T=17`, transition margin `0.7`, and backward temperature `0.8`, PyTorch autograd matches the exact operator prediction to numerical precision (`<1e-12`). The negative mode produces an exactly reversed pressure.

## Relation to prior work

This should not be framed as discovery of Markov spectra. Predictive-state representations, Koopman/transfer-operator methods, and VAMP/VAMPnets all use spectral structure of dynamical systems. Nor should it be framed as discovery of straight-through bias or dead discrete states; both have substantial prior literature.

The narrower methodological contribution is that, in an exactly solvable recurrent predictive-memory system, the source's predictive operator appears *directly and exactly* inside the surrogate gradient field. This gives a controlled bridge between source dynamics, discrete-memory symmetry breaking, and gradient optimization, with no learned probe or sampled estimate in the theorem fixture.
