"""Independent differentiable-memory validation of construction order.

This experiment intentionally lives outside CI. It uses a five-state linear
state-space delay line and PyTorch autograd to audit the local loss/gradient
scaling for dormant downstream prewiring, then records one fixed-step SGD
threshold-crossing reference calculation.
"""

from __future__ import annotations

import math

import numpy as np
import torch


DEPTH = 5
SCALES = np.asarray([0.005, 0.0075, 0.01125, 0.016875, 0.0253125], dtype=float)


def delay_predictions(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(DEPTH, dtype=weights.dtype)
    outputs = []
    for token in inputs:
        next_state = torch.empty_like(state)
        next_state[0] = weights[0] * token
        for index in range(1, DEPTH):
            next_state[index] = weights[index] * state[index - 1]
        state = next_state
        outputs.append(state[-1])
    return torch.stack(outputs)


def delayed_mse(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    predictions = delay_predictions(weights, inputs)
    valid = predictions[DEPTH - 1 :]
    targets = inputs[: valid.numel()]
    return 0.5 * torch.mean((valid - targets) ** 2)


def exact_delayed_mse(weights: torch.Tensor) -> torch.Tensor:
    """Closed form of ``delayed_mse`` for unit-variance binary inputs."""
    gain = torch.prod(weights)
    return 0.5 * (gain - 1.0) ** 2


def binary_input(length: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260830)
    draws = torch.randint(0, 2, (length,), generator=generator, dtype=torch.int64)
    return (2 * draws - 1).to(torch.float64)


def ray_weights(missing: int, scale: float) -> torch.Tensor:
    values = [float(scale)] * missing + [1.0] * (DEPTH - missing)
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def scaling_audit() -> None:
    inputs = binary_input()
    print("order  loss_slope  grad_slope  predicted")
    for missing in range(1, DEPTH + 1):
        improvements = []
        gradient_norms = []
        for scale in SCALES:
            weights = ray_weights(missing, float(scale))
            loss = delayed_mse(weights, inputs)
            improvements.append(float(0.5 - loss))
            loss.backward()
            assert weights.grad is not None
            gradient_norms.append(
                float(torch.linalg.vector_norm(weights.grad[:missing]))
            )

        loss_slope = float(
            np.polyfit(np.log(SCALES), np.log(np.asarray(improvements)), 1)[0]
        )
        grad_slope = float(
            np.polyfit(np.log(SCALES), np.log(np.asarray(gradient_norms)), 1)[0]
        )
        print(
            f"{missing:>5d}  {loss_slope:>10.6f}  {grad_slope:>10.6f}  "
            f"({missing}, {missing - 1})"
        )


def sgd_threshold_steps(
    missing: int,
    *,
    initial_scale: float = 0.05,
    learning_rate: float = 0.05,
    gain_threshold: float = 0.2,
    max_steps: int = 100_000,
) -> tuple[int, float]:
    values = [initial_scale] * missing + [1.0] * (DEPTH - missing)
    weights = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([weights], lr=learning_rate)

    for step in range(max_steps + 1):
        gain = float(torch.prod(weights).detach())
        if gain >= gain_threshold:
            return step, gain
        optimizer.zero_grad(set_to_none=True)
        # For the recurrent binary delay task this is exactly the same objective
        # as ``delayed_mse`` and avoids replaying a sequence tens of thousands of
        # times in the high-order cases.
        loss = exact_delayed_mse(weights)
        loss.backward()
        optimizer.step()
    raise RuntimeError("threshold was not reached")


def optimizer_audit() -> None:
    print("\nfixed-step SGD: init=.05, lr=.05, gain threshold=.2")
    for missing in range(1, DEPTH + 1):
        step, gain = sgd_threshold_steps(missing)
        print(f"order={missing}: step={step}, gain={gain:.9f}")


def exact_formula_check() -> None:
    # The recurrent simulation has gain g=prod_i w_i. With unit-variance binary
    # targets, L=.5(g-1)^2 exactly. This check makes the autograd audit transparent.
    scale = 0.07
    inputs = binary_input()
    for missing in range(1, DEPTH + 1):
        weights = ray_weights(missing, scale)
        loss = delayed_mse(weights, inputs)
        exact_loss = exact_delayed_mse(weights)
        loss.backward()
        gain = scale**missing
        expected_loss = 0.5 * (1.0 - gain) ** 2
        expected_grad_norm = (
            math.sqrt(missing) * (1.0 - gain) * scale ** (missing - 1)
        )
        observed_grad_norm = float(torch.linalg.vector_norm(weights.grad[:missing]))
        if not math.isclose(float(loss), float(exact_loss), rel_tol=1e-12, abs_tol=1e-14):
            raise AssertionError("recurrent and closed-form objectives disagree")
        if not math.isclose(float(loss), expected_loss, rel_tol=1e-12, abs_tol=1e-14):
            raise AssertionError("loss formula mismatch")
        if not math.isclose(
            observed_grad_norm, expected_grad_norm, rel_tol=1e-12, abs_tol=1e-14
        ):
            raise AssertionError("gradient formula mismatch")


if __name__ == "__main__":
    exact_formula_check()
    scaling_audit()
    optimizer_audit()
