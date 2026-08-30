from __future__ import annotations

import math

import numpy as np
import torch


def _delay_predictions(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Run a scalar-input linear delay line with trainable chain gains."""
    depth = int(weights.numel())
    state = torch.zeros(depth, dtype=weights.dtype)
    outputs = []
    for token in inputs:
        next_state = torch.empty_like(state)
        next_state[0] = weights[0] * token
        for index in range(1, depth):
            next_state[index] = weights[index] * state[index - 1]
        state = next_state
        outputs.append(state[-1])
    return torch.stack(outputs)


def _delayed_mse(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    predictions = _delay_predictions(weights, inputs)
    depth = int(weights.numel())
    valid_predictions = predictions[depth - 1 :]
    targets = inputs[: valid_predictions.numel()]
    return 0.5 * torch.mean((valid_predictions - targets) ** 2)


def _base_weights(depth: int, missing: int) -> torch.Tensor:
    values = [0.0] * missing + [1.0] * (depth - missing)
    return torch.tensor(values, dtype=torch.float64)


def _ray_weights(depth: int, missing: int, scale: float) -> torch.Tensor:
    values = [float(scale)] * missing + [1.0] * (depth - missing)
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def _binary_input() -> torch.Tensor:
    # Every valid target has square one, making the population/sample MSE identity
    # exact rather than dependent on a particular random draw.
    return torch.tensor(
        [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        dtype=torch.float64,
    )


def test_dormant_downstream_initialization_preserves_current_predictor() -> None:
    inputs = _binary_input()
    depth = 5

    reference_predictions = None
    for missing in range(1, depth + 1):
        weights = _base_weights(depth, missing)
        predictions = _delay_predictions(weights, inputs)
        loss = _delayed_mse(weights, inputs)

        assert torch.allclose(
            predictions, torch.zeros_like(predictions), atol=0.0, rtol=0.0
        )
        assert float(loss) == 0.5
        if reference_predictions is None:
            reference_predictions = predictions
        else:
            assert torch.equal(predictions, reference_predictions)


def test_autograd_matches_exact_construction_order_formulas() -> None:
    inputs = _binary_input()
    depth = 5
    scale = 0.07

    for missing in range(1, depth + 1):
        weights = _ray_weights(depth, missing, scale)
        loss = _delayed_mse(weights, inputs)
        improvement = 0.5 - loss
        loss.backward()

        gain = scale**missing
        expected_improvement = gain - 0.5 * gain * gain
        expected_missing_gradient_norm = (
            math.sqrt(missing) * (1.0 - gain) * scale ** (missing - 1)
        )

        assert math.isclose(
            float(improvement), expected_improvement, rel_tol=1e-11, abs_tol=1e-14
        )
        assert weights.grad is not None
        observed_norm = float(torch.linalg.vector_norm(weights.grad[:missing]))
        assert math.isclose(
            observed_norm,
            expected_missing_gradient_norm,
            rel_tol=1e-11,
            abs_tol=1e-14,
        )


def test_loss_and_gradient_scaling_recover_orders_one_through_five() -> None:
    inputs = _binary_input()
    depth = 5
    scales = np.asarray([0.01, 0.015, 0.0225, 0.03375], dtype=float)

    for missing in range(1, depth + 1):
        improvements = []
        gradient_norms = []
        for scale in scales:
            weights = _ray_weights(depth, missing, float(scale))
            loss = _delayed_mse(weights, inputs)
            improvements.append(float(0.5 - loss))
            loss.backward()
            assert weights.grad is not None
            gradient_norms.append(
                float(torch.linalg.vector_norm(weights.grad[:missing]))
            )

        loss_slope = float(
            np.polyfit(np.log(scales), np.log(np.asarray(improvements)), 1)[0]
        )
        gradient_slope = float(
            np.polyfit(np.log(scales), np.log(np.asarray(gradient_norms)), 1)[0]
        )

        assert abs(loss_slope - missing) < 0.04
        assert abs(gradient_slope - (missing - 1)) < 0.04


def test_exact_zero_initialization_moves_only_first_order_case() -> None:
    inputs = _binary_input()
    depth = 5

    for missing in range(1, depth + 1):
        weights = _base_weights(depth, missing).requires_grad_(True)
        loss = _delayed_mse(weights, inputs)
        loss.backward()
        assert weights.grad is not None
        missing_norm = float(torch.linalg.vector_norm(weights.grad[:missing]))
        if missing == 1:
            assert missing_norm > 0.0
        else:
            assert missing_norm == 0.0
