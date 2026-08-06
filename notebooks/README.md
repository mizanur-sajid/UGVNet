# Training notebooks

The two notebooks use the same UGVNet architecture, dataset audit, training
loop, metrics, evaluation, and inference code.

- `ugvnet_colab.ipynb` uses Drive-only project storage under
  `/content/drive/MyDrive/SkinDisNet/ugvnet`, including dataset and Torch caches.
- `ugvnet_kaggle.ipynb` uses Kaggle `Add Input` and writes directly under
  `/kaggle/working/ugvnet` with higher local-disk concurrency.

The runtime introduction, data configuration, dataset resolver, and platform
notes are intentionally native to each service. Keep the shared architecture,
audit, training, evaluation, and inference sections synchronized.

Colab persists artifacts in the Drive project's `colab/` folders. Kaggle writes
directly to native folders under `/kaggle/working/ugvnet`. Cell 20 exports CSV
files and creates a timestamped report plus a stable `latest` PDF copy. Cell 14 opens the TensorBoard dashboard before training.


## Automatic performance and generalization control

Automatic mode treats a training split with at least 8,000 images as large.
The large profile keeps EfficientNetV2-S and ConvNeXt-Tiny, while reducing the
input to 224px and the fusion head to 256 channels with one refinement block.
It also uses a two-epoch frozen-backbone warm-up, channels-last CUDA,
prefetched loaders, larger evaluation batches, and GPU-side metric accumulation.

Every completed epoch counts. Starting at epoch 3, a train–validation accuracy
gap above 10 percentage points is tracked as a warning only. Two consecutive
high-gap epochs trigger a next-epoch adjustment: learning rates are multiplied
by 0.70, dropout and label smoothing rise gradually within configured limits,
and focal gamma decreases when focal loss is active. No checkpoint rollback,
epoch retry, rejected-history file, or gap-based stop remains.

## Output theme and speed defaults

Final output shows Best Training Accuracy in %, Best Validation Accuracy in %,
and Best Test Accuracy in %, with decimal values and the relevant
epoch. Notebook HTML automatically switches to light text on dark backgrounds
and dark text on light backgrounds. Confusion matrices show exact image counts with
automatic black/white annotation contrast in both notebook output and PDF reports.

The audit reads every image exactly once and reuses the bytes for decoding and
SHA-256 duplicate detection. Training uses four-batch DataLoader prefetching,
channels-last CUDA tensors, mixed precision, GPU-side metric accumulation, and
fused AdamW when supported. Kaggle and Colab use the same model and speed logic;
only their storage and dataset-source cells differ.



## Class imbalance and continuation

The training notebooks use effective-number sampling instead of full inverse
frequency by default. Classes containing no more than half as many images as the
largest class receive a stronger transform when imbalance handling is active.
Smoothed cross-entropy remains the normal loss; focal loss is selected
automatically only at an imbalance ratio of 3× or greater.

Set `RESUME_CHECKPOINT_PATH` in Cell 04 before running Cell 13 to continue a
trusted project checkpoint. The complete optimization, early-stopping,
regularization, freeze, history, and random-number state is restored. Cell 19
creates class-error and confusion-pair CSV files, and Cell 20 includes them in
the PDF report.

## TensorBoard

TensorBoard is enabled by default and writes one timestamped directory per run. Cell 13 logs an expandable operation graph with the outer model scope promoted into named architecture sections. It also writes `Model/full_architecture` to Images and saves `ugvnet_full_architecture.png` in the platform results directory. The image shows both backbone stages, adaptive gating, every configured global-fusion block, normalization, pooling, dropout, classifier, executed shapes, and parameter counts.
Colab persists logs under the Drive project's `tensorboard/colab/` folder;
Kaggle writes under `/kaggle/working/ugvnet/tensorboard/`.

Every completed epoch records training/validation metrics, per-class precision,
recall, specificity and ROC-AUC, calibration error, hybrid-fusion behavior,
learning rates, generalization decisions, throughput, and system/GPU telemetry.
Epoch 1 and then every five completed epochs add selected activations, AdamW
moment distributions, and sampled validation embeddings. Parameter and gradient
histograms retain their performance-aware interval.

Cell 19 adds test PR and ROC curves, a reliability diagram, prediction and fusion
images, fused-feature Grad-CAM, numbered confusion matrix, final embeddings,
complete class metrics, class-wise error rates, top confusion pairs, HParams, and checkpoint/report paths. Full-model
histograms are available with `TENSORBOARD_LOG_FULL_MODEL_HISTOGRAMS = True`,
but remain off by default to protect large-dataset speed.

The Profiler UI plugin is installed on a best-effort basis. If package downloads
are unavailable, training continues and the raw trace is still written. Audio
and Mesh are intentionally not logged because they are not meaningful outputs
for this image classifier.
