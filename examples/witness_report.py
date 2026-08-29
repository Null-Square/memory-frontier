from memory_frontier import (
    alias_entropy,
    bayes_log_loss,
    best_deterministic_controller,
    best_recursive_quotient,
    best_static_partition,
    four_state_aliasing_witness,
    source_stationary_distribution,
)

source = four_state_aliasing_witness()
static = best_static_partition(source, 2)
online = best_deterministic_controller(source, 2)
quotient = best_recursive_quotient(source, 2)

print("stationary:", source_stationary_distribution(source))
print("bayes_nll:", bayes_log_loss(source))
print("static_k2:", static.loss, static.assignment)
print("deterministic_online_k2:", online.loss)
print("optimal_transition_table:\n", online.transition_table)
print("optimal_initial_memory:", online.initial_memory)
print("stationary_source_memory_occupancy:\n", online.occupancy)
print("alias_entropy_H(M|S):", alias_entropy(online.occupancy))
print("recursive_quotient_k2:", quotient.loss, quotient.assignment)
print("online_tax:", online.loss - static.loss)
print("adaptive_advantage:", quotient.loss - online.loss)
