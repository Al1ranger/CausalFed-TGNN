# Observed results

230 completed recorded fits. No statistical significance claim. Mean ± sample SD requires five valid seeds.

- 50,000 events: full variant known AUROC 0.4903 ± 0.0221, AP 0.0211 ± 0.0013, F1 0.0401 ± 0.0050 (n=5).
- 100,000 events: full variant known AUROC 0.5037 ± 0.0151, AP 0.0210 ± 0.0026, F1 0.0351 ± 0.0062 (n=5).
- 500,000 events: full variant known AUROC 0.5061 ± 0.0137, AP 0.0217 ± 0.0007, F1 0.0373 ± 0.0020 (n=5).

The study remains partial: finite-history architecture, fixed training budget, no unseen-bank transfer, no public-data validation, and no continual or realistic burst-timing experiment. Interpret comparisons using the history-matched MLP and scale-specific paired ablations.

## Matched observed comparisons

- 50,000: history MLP AUROC 0.4888 ± 0.0295; full variant 0.4903 ± 0.0221. No consistent advantage of the full variant across these conditions is observed.
  - Unknown amount: AUROC 0.5367 ± 0.0104; AP 0.0463 ± 0.0029; FPR95 0.9532 ± 0.0070.
  - Unknown energy: AUROC 0.4997 ± 0.0238; AP 0.0404 ± 0.0043; FPR95 0.9499 ± 0.0157.
  - Unknown negative_margin: AUROC 0.4941 ± 0.0319; AP 0.0384 ± 0.0035; FPR95 0.9482 ± 0.0183.
  - Unknown prototype_distance: AUROC 0.5079 ± 0.0295; AP 0.0407 ± 0.0029; FPR95 0.9406 ± 0.0106.
- 100,000: history MLP AUROC 0.4865 ± 0.0075; full variant 0.5037 ± 0.0151. No consistent advantage of the full variant across these conditions is observed.
  - Unknown amount: AUROC 0.5379 ± 0.0142; AP 0.0460 ± 0.0022; FPR95 0.9492 ± 0.0056.
  - Unknown energy: AUROC 0.5763 ± 0.0348; AP 0.0542 ± 0.0052; FPR95 0.9318 ± 0.0168.
  - Unknown negative_margin: AUROC 0.5809 ± 0.0168; AP 0.0538 ± 0.0011; FPR95 0.9313 ± 0.0127.
  - Unknown prototype_distance: AUROC 0.5253 ± 0.0262; AP 0.0406 ± 0.0029; FPR95 0.9270 ± 0.0252.
- 500,000: history MLP AUROC 0.5088 ± 0.0099; full variant 0.5061 ± 0.0137. No consistent advantage of the full variant across these conditions is observed.
  - Unknown amount: AUROC 0.5441 ± 0.0054; AP 0.0463 ± 0.0018; FPR95 0.9425 ± 0.0020.
  - Unknown energy: AUROC 0.6353 ± 0.0192; AP 0.0631 ± 0.0057; FPR95 0.8683 ± 0.0293.
  - Unknown negative_margin: AUROC 0.6469 ± 0.0072; AP 0.0672 ± 0.0021; FPR95 0.8663 ± 0.0118.
  - Unknown prototype_distance: AUROC 0.5061 ± 0.0131; AP 0.0382 ± 0.0021; FPR95 0.9409 ± 0.0107.
- 50k clean fedavg: known AUROC 0.4951 ± 0.0559; AP 0.0217 ± 0.0027; F1 0.0425 ± 0.0046.
- 50k clean fedprox: known AUROC 0.4951 ± 0.0559; AP 0.0217 ± 0.0027; F1 0.0425 ± 0.0046.
- 50k clean median: known AUROC 0.4967 ± 0.0603; AP 0.0224 ± 0.0036; F1 0.0440 ± 0.0051.
- 50k clean scaffold: known AUROC 0.4951 ± 0.0559; AP 0.0217 ± 0.0026; F1 0.0429 ± 0.0054.
- fedavg/label_flip: paired clean-minus-attacked AUROC 0.0128 ± 0.0093. Negative degradation means the attacked run scored higher; no robustness guarantee follows.
- fedavg/poison: paired clean-minus-attacked AUROC 0.0145 ± 0.0937. Negative degradation means the attacked run scored higher; no robustness guarantee follows.
- fedavg/sybil: paired clean-minus-attacked AUROC 0.0159 ± 0.1034. Negative degradation means the attacked run scored higher; no robustness guarantee follows.
- median/label_flip: paired clean-minus-attacked AUROC 0.0156 ± 0.0109. Negative degradation means the attacked run scored higher; no robustness guarantee follows.
- median/poison: paired clean-minus-attacked AUROC 0.0187 ± 0.0118. Negative degradation means the attacked run scored higher; no robustness guarantee follows.
- median/sybil: paired clean-minus-attacked AUROC 0.0161 ± 0.1027. Negative degradation means the attacked run scored higher; no robustness guarantee follows.