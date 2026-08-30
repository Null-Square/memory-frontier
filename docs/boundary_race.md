# Boundary geometry predicts the next hard algorithm

The adiabatic surrogate map uses only the current hard transition table. Exact training trajectories show that this is not enough: the continuous transition-logit geometry inherited on entry to a hard cell strongly affects which argmax boundary is reached next.

For a current hard target `j` and alternative target `n`, define

\[
d_{j\to n}=z_j-z_n>0
\]

and the current gradient-descent pressure

\[
p_{j\to n}=g_j-g_n.
\]

Positive `p` means an SGD step shrinks the margin. Under a frozen-gradient linearization with transition learning rate `eta`, the estimated crossing time is

\[
\boxed{\hat\tau_{j\to n}=\frac{d_{j\to n}}{\eta p_{j\to n}}.}
\]

`predict_boundary_race` evaluates every positive-pressure one-edit candidate and predicts the smallest `hat_tau`.

This is deliberately a local approximation: decoder logits and the transition softmax geometry continue to evolve while the model remains inside the cell. It is therefore expected to predict edge identity better than exact residence time.

## Exact-SGD population result

Using exact expected gradients, joint transition/readout SGD, uniform initial readouts, and the delayed-repeat family, the boundary-race predictor was evaluated at hard-cell entry.

| Delay | K | single-edit events | entry prediction coverage | accuracy when covered | overall edge accuracy | adiabatic successor accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 458 | 94.1% | 98.6% | 92.8% | 38.6% |
| 3 | 4 | 610 | 96.6% | 97.5% | 94.1% | 48.7% |
| 4 | 5 | 499 | 98.0% | 99.4% | 97.4% | 47.5% |

The precise counts come from fixed scratch seeds with 30 runs for delays 2 and 3 and 15 runs for delay 4. The repository experiment exposes the same protocol for reproduction and extension.

Across several learning-rate ratios on delay 2, conditional next-edge accuracy remained roughly `94-99%`. Cases without an entry-time positive-pressure candidate are expected: decoder evolution can create a useful transition pressure later in the residence interval.

Residence-time prediction is weaker but still informative. For correctly predicted edges, log predicted versus log actual residence time had correlations around `0.74-0.81`; the naive frozen-gradient estimate typically underestimates the residence time by about a factor of two because the pressure evolves after entry.

## Consequence

The same hard controller can have different successors depending on continuous logit margins. Therefore the optimization process is not Markov when coarse-grained to hard tables alone:

\[
F_t \not\Rightarrow P(F_{t+1})\ \text{deterministically}.
\]

The relevant hybrid state includes at least

\[
(F_t,z_t,a_t)
\]

and, for momentum/adaptive optimizers, optimizer state as well.

This clarifies the relationship between two exact approximations in the project:

- **adiabatic surrogate flow** captures which edit is preferred after decoder equilibration and canonicalized cell geometry;
- **boundary race** captures which edit is physically closest under the actual continuous state currently carried by training.

The large accuracy gap between them demonstrates that inherited boundary geometry is a first-order determinant of the next learned algorithm, not a minor implementation detail.
