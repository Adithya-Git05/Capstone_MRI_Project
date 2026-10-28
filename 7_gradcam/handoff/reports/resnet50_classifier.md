# ResNet50 classifier

**Status: implemented with a partial checkpoint, not a completed/evaluated
experiment.** The model replaces the standard ResNet50 stem with a
one-channel `Conv2d(1,64,7,stride=2,padding=3,bias=false)` and retains
BatchNorm/ReLU/MaxPool; its head is `Linear(2048,4)`. It uses
`weights=None` (not pretrained). Source:
[`model.py`](../../../4_2d_classification/src/classification/model.py#L7).

The local checkpoint
[`best.pt`](../../../4_2d_classification/outputs/classification/20261020_163432_seed2026/best.pt)
is 94,340,781 bytes, SHA-256
`800BFBAA39B2216E40A4455B0085CE1CA483CEB7206BA87C42835C00918C5EDB`. It
contains model-state/configuration metadata and no optimizer state. Its log
contains only two epochs:

| Epoch | Train loss | Validation loss | Validation accuracy | Learning rate |
|---:|---:|---:|---:|---:|
| 1 | 0.90735 | 1.25281 | 0.7637 | 0.001 |
| 2 | 0.60688 | 0.64464 | 0.8131 | 0.0009972637 |

The configured protocol is 160x160 single-channel BRISC input, batch 32,
seed 2026, up to 30 epochs, cross-entropy, AdamW (lr 0.001, weight decay
0.0001), cosine scheduler (eta_min 1e-6), patience 7, and no pretrained
weights. Metadata says `checkpoint:null` and `metrics:null` despite the local
file. No test metrics, confusion matrix, or ROC/PR values are available.
Source files: run
[`config.yaml`](../../../4_2d_classification/outputs/classification/20261020_163432_seed2026/config.yaml),
[`run.log`](../../../4_2d_classification/outputs/classification/20261020_163432_seed2026/run.log),
[`metadata.json`](../../../4_2d_classification/outputs/classification/20261020_163432_seed2026/metadata.json).

Grad-CAM target: `model.network.layer4`, hook shape `[B,2048,5,5]` for
160x160 input. This target was observed in a forward hook. Use a selected raw
class logit. Grad-CAM itself was not tested.

Recover the remaining run history and a completed official-test evaluation
before reporting this as a trained result.
