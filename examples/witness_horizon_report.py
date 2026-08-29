from memory_frontier.finite import best_finite_horizon_deterministic_controller
from memory_frontier.witnesses import four_state_aliasing_witness

source = four_state_aliasing_witness()
for horizon in (1, 2, 3, 4, 8, 16, 32, 64, 128):
    optimum = best_finite_horizon_deterministic_controller(source, 2, horizon)
    print(
        f"T={horizon:3d}  loss={optimum.loss:.10f}  m0={optimum.initial_memory}  "
        f"F={optimum.transition_table.tolist()}"
    )
