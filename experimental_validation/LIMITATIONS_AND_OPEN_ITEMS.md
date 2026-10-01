# Limitations and open items

- Screening runs use eight epochs/rounds without a hyperparameter search or a
  convergence claim. Model superiority is not established by completing a run.
- The finite-history, one-layer snapshot implementation narrows the conceptual
  manuscript architecture. It does not implement arbitrary temporal memory,
  end-to-end history aggregation, or multi-hop dynamic shared entity states.
- Same four synthetic banks appear in training and testing. Invariance ablations
  measure same-bank temporal performance, not unseen-bank generalization.
- Graph inputs contain additional entity histories. History MLP controls access
  to those inputs; Context MLP alone cannot isolate the value of graph structure.
- Seeds vary both generator realizations and optimization. Five seeds provide
  limited uncertainty information and no population-level significance claim.
- Original daily bank spacing remains unchanged. Entity revisit gaps vary but
  do not validate a variable event-arrival or burst-timing mechanism.
- Known-class results exclude D/E. Unknown results use all test events and a
  separate validation-only threshold. Neither population may be silently changed.
- Federated execution is a single-machine simulation with pooled validation.
  Local IRM and cross-bank contrastive terms are disabled; no differential privacy,
  secure aggregation, causal identification, or formal robustness is implemented.
- Adaptive aggregation has no existing concrete rule. It remains PENDING.
- Continual replay, drift adaptation, temporal-stress-v2, public-data empirical
  evaluation, and explanation-quality evaluation remain PENDING until separately
  executed. Their absence does not invalidate completed, accurately scoped runs.
- A PaySim partial download is not admissible evidence. IEEE-CIS requires access
  and acceptance of applicable terms. No public-data performance is inferred.
- Timing and memory scopes are defined in EXPERIMENTAL_METHODS.md. Lifetime memory
  peaks must not be interpreted as isolated model resource requirements.
- The requested filename ending in Complete does not establish scientific
  completion. The minimum success gate must be assessed from actual artifacts.
