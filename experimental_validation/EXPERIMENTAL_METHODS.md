# Experimental methods

## Scope and study design
This is an eight-epoch screening study of a newly implemented finite-history
CausalFed-TGNN variant. It is not a claim that the complete conceptual framework
has been validated. The original 15 streams are immutable. The experimental unit
for uncertainty reporting is a generator/training seed, 42–46, at each scale.
No hyperparameter search, causal identification, privacy mechanism, public-data
validation, or unseen-institution evaluation is implied.

## Populations and preprocessing
Within each bank, sort by timestamp and transaction ID. For rank i among n events,
use i/(n−1)<0.5 for training, [0.5,0.7) for validation, and ≥0.7 for testing.
Assertions verify globally separated partition timestamps and exclude D/E from
training and validation. Known classification evaluates legitimate and A/B/C test
events; the separate all-late population includes D/E. Novelty treats D/E as the
positive class and every other late event as negative.

The six transaction inputs reproduce the previous Context LR feature set:
log(1+amount), log(1+1000 times historical merchant count/(bank rank+1)), the
analogous device count, and three bank indicators. Standardization statistics
come exclusively from training data. No labels, mechanism tags, periods,
label-availability timestamps, full-period entity degrees, or future identities
are predictive inputs. The original zero label delay is assumed in fitting.

## Graph definition
Each prediction has one transaction node and four entity snapshot nodes: account,
merchant, device, and location. Account and customer are redundant one-to-one
aliases in the supplied generator; customer is not separately duplicated. All
entity identifiers are scoped to a bank, including superficially shared strings.
Each entity points to its transaction with `context_for`; reverse `uses` edges
are also constructed. There are no edges between different prediction snapshots.
This restriction prevents a batch from sharing a future entity state.

Before appending event i, the snapshot contains: mean log amount of at most eight
previous same-entity events; log(1+1000 times prior entity count/(bank rank+1));
log(1+elapsed days since last same-entity event); and a prior-history indicator.
Unseen entities have all-zero raw snapshots. Snapshot normalization uses only the
training partition. Earlier unlabeled validation/test events update histories
prequentially; their labels never enter features. Original bank event gaps are
fixed at one day, although same-entity revisit gaps vary. These data cannot test
realistic burst timing or isolate a variable event-arrival mechanism.

The summary aggregation is deterministic, followed by trainable message passing.
This is not an unrestricted end-to-end temporal-memory model or a full shared
entity graph. The one-layer encoder means reverse outputs do not influence the
current transaction readout. Parameter counts include those registered weights.

## Model family
MLP: six transaction inputs, two ReLU layers of width 24, binary logit head.
History MLP: identical form with all 16 standardized entity snapshot values added;
this controls access to historical information without graph message passing.
GraphSAGE: one homogeneous SAGEConv on entity-to-transaction and reverse edges,
shared entity projection, residual transaction projection, ReLU, binary head.
Temporal GraphSAGE adds the entity elapsed-time column. The static variant zeros
that column but still uses causal historical snapshots; it is not a global
transductive static-graph benchmark.
HGT: one PyG HGTConv, width 24, two heads, distinct entity types and relations,
residual transaction projection, ReLU, binary head. Heterogeneous static uses the
same HGT encoder without elapsed-time input.
CausalFed-TGNN variant: HGT plus width-24 invariant and environment branches.
The fraud head uses only the invariant branch; an auxiliary four-bank head uses
the environment branch. The homogeneous ablation retains HGT attention but merges
all entity types and relations, sharing the entity projection. Its parameter count
changes and is reported. No claim that architectures have equal capacity is made.

HGT reference: https://arxiv.org/abs/2003.01332
Operator documentation: https://pytorch-geometric.readthedocs.io/en/stable/tutorial/heterogeneous.html

## Objective and ablations
L = mean binary cross entropy + 0.1 L_IRM + 0.01 L_con + 0.1 L_env.
L_IRM is the mean over present banks of the squared derivative with respect to
a scalar a of CE(a times logits, y), evaluated at a=1 (IRMv1-inspired).
L_con is supervised contrastive loss at temperature 0.2 on the first at most 128
randomly shuffled batch examples: positives have the same class and a different
bank; all other examples except self are in the denominator. Anchors without a
cross-bank positive are excluded. L_env is four-way bank cross entropy.
Replay and KL coefficients are zero. IRM and contrastive loss are disabled and
logged when fewer than two environments are present. No_inv, no_env, no_time,
no_con, and homogeneous variants change only their documented components.
The open-set-module ablation is a post hoc scoring comparison: negative logit
margin supplies a classifier-only comparator without retraining a duplicate
network. It does not remove a trained novelty loss because none is used.

## Training and selection
All models use seed-matched Python/NumPy/PyTorch RNGs, deterministic PyTorch CPU
algorithms, two CPU threads, batch size 2048, eight complete epochs, Adam at 0.001,
and unweighted cross entropy. Training batches are randomly permuted. No test
metric chooses a model, epoch, threshold, or loss coefficient. Select the checkpoint
with lowest validation cross entropy, resolving ties by the earlier checkpoint.
Eight epochs is a screening budget, not evidence of convergence or optimal tuning.
Class imbalance may make this budget and objective inadequate; negative results
must be interpreted with that limitation. The same seed is used for dataset and
initialization, so their variance contributions are not separately estimated.

## Open-set scoring and thresholds
Energy is −logsumexp of the two logits, with temperature 1. Negative absolute
logit difference is the maximum-logit-margin uncertainty score. Prototype distance
is the minimum Euclidean distance in the final fraud representation to the two
class centroids, calculated exclusively from training embeddings. An amount-only
novelty diagnostic is evaluated on exactly the same held-out population.
All scores increase toward unknown. No score normalization is applied. Novelty
thresholds are the linear 95th percentile of known validation scores; predict
unknown at score ≥ threshold. Ties can produce a false alarm rate above 5%.
The known-class threshold maximizes validation F1, with highest-threshold tie
breaking. These two thresholds are independent. FPR95 is the first empirical
test ROC point with TPR ≥0.95; it is a ranking diagnostic, not a deployable
threshold calibrated from unknown test labels. AUROC and AP use tied-score-aware
sklearn implementations. AP is not trapezoidal PR area. Brier score and ten
equal-width-bin ECE apply to fraud probabilities only. Single-class metrics
requiring both labels return null, never an invented value.

## Federated simulation
Four local banks participate in every round. Eight rounds each contain one local
epoch. All clients start from the same global weights and use SGD at 0.03.
FedAvg and FedProx average weights by training sample count; FedProx adds
0.01/2 times squared distance from the broadcast weights. SCAFFOLD uses correction
c−c_i in local gradients and updates c_i = c_i−c+(x−y_i)/(K_i η); server c is the
mean of four controls under full participation. Global model learning rate is 1.
Coordinate median averages the two middle values with four clients. There is no
defined adaptive weighting algorithm in the existing repository, so that remains
pending rather than inventing a method under its name.
Each local batch has one bank; IRM and cross-bank contrastive losses are disabled.
Thus this is a federated architectural variant, not the pooled invariant objective.
The server selects checkpoints and thresholds using pooled validation in this
simulation. It is not a private deployment protocol, nor a distributed network
benchmark. The centralized/federated comparison also changes optimizer and
optimization schedule, so cannot isolate data decentralization alone.

SCAFFOLD reference: https://proceedings.mlr.press/v119/karimireddy20a.html

## Controlled attacks
If executed, bank 0 is malicious from round 1. Label flipping independently flips
50% of its local training labels per epoch. Update poisoning sends −5 times its
honest update. Sybil/collusion submits that poisoned update under three identities
(the original plus two clones), yielding three malicious identities among six;
the fraction of malicious underlying banks remains 1/4. Attackers know their own
data and broadcast parameters and target untargeted degradation. Clean/attacked
runs share seeds and partitions; label-flipping RNG draws can change subsequent
shuffles. No backdoor-success claim is made without an implemented trigger.

## Provenance, uncertainty, and costs
Each completed run records immutable dataset and split hashes, source snapshot,
configuration, selected checkpoint, predictions, round/epoch logs, package versions,
hardware, and result hashes. A source hash replaces an unavailable Git commit;
no commit identifier is fabricated. Checkpoint reload equality is asserted on 128
held-out examples. Summary means and sample SD (ddof=1) are reported only for five
completed distinct seeds; run counts and failures remain visible. Paired ablation
and attack differences use identical scale/seed populations. No p-values or
confidence intervals are implied.
Train timing includes setup and validation passes; inference timing includes
training prototype construction, validation/test prediction and metric computation.
Peak CPU memory is the process lifetime high-water mark, not isolated per-model
peak. Federated communication is a calculated tensor-payload count, excluding
transport overhead; it is not measured network traffic. Costs must retain these
definitions in any table or manuscript claim.
