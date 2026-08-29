# Forward-identical initialization, different learnability

The dormant-scaffold theorem produces a deterministic optimization experiment in
which two models start with exactly the same forward computation and prediction
loss but follow different training trajectories solely because their unreachable
memory states contain different hard transition structure.

## Protocol

Use the binary delay-3 repeat source with switch probability `rho=0.1`. Both
models have `K=4` memory states, start in memory 0, use the same decoder logits,
and have the same reachable transition row

```text
F(0,0)=0
F(0,1)=0
```

so every forward trajectory remains in memory 0 initially. The shared decoder is
uniform in states 0, 1, and 2; state 3 has fixed logits `(0.2,-0.2)`. Since state
3 is unreachable, the two models make identical predictions and both start at

\[
L_0=\log 2=0.69314718056.
\]

The only difference is the unreachable hard topology.

**Inert initialization**

```text
1 -> 0
2 -> 0
3 -> 0
```

for either input symbol.

**Delay-matched scaffold**

```text
1 -> 2 -> 3 -> 0
```

again independent of the input symbol. The scaffold is never executed in the
hard forward pass at initialization.

Transition logits use margin `0.7` and backward temperature `0.8`. Decoder logits
are frozen. We run plain gradient descent on transition logits only, using the
NumPy exact-gradient oracle rather than sampled sequences or autograd:

```text
learning rate = 5.0
horizon       = 20
steps         = 80
```

## Result

The inert model never changes its hard forward controller in this protocol and
remains at

```text
loss = 0.69314718056
```

through all 80 updates.

The scaffolded model changes the first reachable transition at step 7:

```text
before: F(0,0)=0, F(0,1)=0
step 7: F(0,0)=1, F(0,1)=0
```

and its exact fixed-readout loss immediately drops to approximately

```text
0.66968255354
```

A later transition change further lowers the loss to approximately

```text
0.65815206549.
```

The regression test uses only the robust part of this observation: after 20
identical updates, the inert model is still forward-collapsed while the
scaffolded model has activated the predicted token-0 transition and improved
exact loss by more than `0.02` nats/token.

## Interpretation

This is stronger than ordinary initialization sensitivity. At step zero the two
models have the same:

- architecture and memory cardinality;
- initial memory state;
- reachable hard transition function;
- decoder parameters;
- forward memory trajectory on every possible sequence;
- prediction distribution on every possible sequence;
- exact expected loss.

They differ only in hard transitions belonging to states that are behaviorally
unreachable. The hard-forward straight-through backward pass nevertheless sends
counterfactual derivative mass through those states. A delay-matched dormant
chain exposes the lag-3 predictive dependency and creates a useful gradient;
the inert unreachable topology does not.

This should be positioned against two established literatures rather than
presented as a generic discovery of finite-state gradients. Exact/model-based
policy gradients for finite-state controllers are classical, and differentiable
finite/weighted automata have been trained by gradient methods. The narrower
phenomenon here comes from the hard-forward / soft-backward mismatch: classical
automata can delete unreachable states without changing behavior, while in this
STE parameterization changing such states can change the training vector field.
