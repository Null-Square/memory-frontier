# Readout initialization is an optimization resource

A hard categorical memory state can be unused by the current controller. Its
readout is then **not identified by data**. Any value assigned to that unused
readout is therefore an extra optimization convention, not part of the hard
finite-state prediction problem.

This matters for straight-through training because the decoder values determine
which transition edits receive gradient pressure.

## Collapsed-controller stationary point

Let the source begin in stationarity and let

\[
p_X(x)=P(X_{t+1}=x)
\]

be its stationary one-token marginal. Consider a deterministic controller whose
memory is collapsed to one state for the entire scored horizon. Initialize every
memory state's readout to the same source marginal:

\[
q_m(x)=p_X(x) \quad \forall m.
\]

Then the joint trainable system is stationary under the exact-distribution
straight-through objective:

1. the occupied readout is already the optimal predictor available to a memory
   state that carries no source information;
2. unused readouts receive no data gradient; and
3. because every readout is identical, redirecting probability mass to another
   memory target has no first-order effect on prediction loss, so the transition
   gradient is zero.

The repository freezes this fact as a numerical regression using double
precision and the exact expected finite-horizon objective.

## Why uniform initialization is not neutral

If all readouts instead begin uniform while the source marginal is non-uniform,
the occupied readout moves toward the source marginal while an unused readout
stays at its arbitrary initial value. The resulting decoder disagreement creates
transition pressure that can revive the unused state.

Thus a nominally innocuous decoder initialization can act as an **exploration
prior** for the discrete state transition learner.

`readout_prior.py` therefore makes the convention explicit:

- `uniform`
- `source_marginal`
- `random`

The original learner remains unchanged for historical comparison; new ablations
should name the readout initialization explicitly.

## Interpretation boundary

Unused or dead discrete codes are a known issue in vector quantization, and many
methods use reinitialization, stochastic assignments, or codebook updates to
restore utilization. Recent examples include work on codebook collapse and
non-stationary vector quantization (e.g. arXiv:2602.18896 and arXiv:2606.11363).

The research question here is narrower: because the finite-memory prediction
objective is exactly enumerable, we can distinguish representational capacity,
hard-controller optimality, and the optimization effect of an otherwise
unidentified decoder prior. We do **not** claim dead-code phenomena themselves
as novel.

## Reproduce the pilot ablation

```bash
python -m pip install -e '.[dev,optimization]'
python examples/readout_prior_report.py
```

Do not treat a recovery-rate difference as a theorem. The theorem-level claim is
the collapsed source-marginal stationary point; recovery rates are optimizer and
protocol dependent empirical measurements.
