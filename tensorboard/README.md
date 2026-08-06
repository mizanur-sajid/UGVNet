# TensorBoard logs

Colab runs are stored in `colab/<timestamped_run>/`. Kaggle logs remain inside
`/kaggle/working/ugvnet/tensorboard` and are not copied into Drive. Generated
event files are ignored by Git.

Run Cell 14 before training. The updated notebooks provide:

- Scalars for audit, train/validation/test, class metrics, calibration,
  hybrid-fusion internals, learning rates, non-blocking gap warnings, regularization adaptations, runtime loss settings, and system/GPU usage.
- Custom Scalars with organized dashboards: UGVNet accuracy, quality, throughput,
  hybrid fusion diagnostics, learning rates, system resources, and runtime
  adaptation settings — all plotted as grouped multiline charts.
- A promoted, expandable operation graph organized into the two backbones, adaptive fusion, global refinement, normalization, pooling, dropout, and classifier; expand any section in Graphs to inspect its traced operations.
- A high-resolution `Model/full_architecture` image with real stage shapes and parameter counts, also saved to `results/ugvnet_full_architecture.png`.
- Text, selected parameter/gradient/activation/optimizer distributions, and a five-batch CPU/CUDA profiler trace.
- Validation embedding evolution plus final Projector embeddings.
- Prediction, fusion, confusion, ROC, reliability, and Grad-CAM images.
- Per-class PR curves and error-rate scalars, a Text table of the largest true/predicted confusion pairs, HParams, and artifact/checkpoint paths.
- Cross-reference to the Netron-viewable ONNX model export under `Model/netron_export`.

Heavy diagnostics are sampled on epoch 1 and then at configurable intervals.
Exhaustive model histograms are opt-in. Refresh or rerun Cell 14 after Cell 19
to discover final plugins. Existing event files cannot acquire new dashboards;
start a new timestamped run after notebook updates.

## T4 memory safety

When a T4 GPU (16 GB VRAM) is detected, the notebook automatically caps:
- `TENSORBOARD_MAX_EMBEDDING_SAMPLES` to 256 (from 512)
- `TENSORBOARD_LOG_FULL_MODEL_HISTOGRAMS` to `False`
- `TENSORBOARD_MAX_GRADCAM_IMAGES` to 4
- Profiler never uses `with_stack=True`
- `torch.cuda.empty_cache()` is called after every heavy TensorBoard operation

These caps keep TensorBoard fully functional without risking OOM crashes.

## Profiler plugin

The notebook tries two profiler UI plugins in order:
1. `torch-tb-profiler` — the PyTorch-maintained plugin
2. `tensorboard-plugin-profile` — the TensorFlow-maintained plugin

If neither installs, all other TensorBoard dashboards still work and the
profiler trace JSON is still saved for offline analysis.

## Netron alongside TensorBoard

Cell 14.b exports the model to ONNX (with TorchScript fallback) and displays
it interactively in an iframe. The ONNX path is logged to TensorBoard under
`Model/netron_export` for cross-referencing. Netron and TensorBoard run
concurrently on separate ports without conflicts.

## Refreshing after training

Cell 14.c (inserted after test evaluation) kills and relaunches TensorBoard so
that all post-training plugins are immediately visible — HParams, PR curves,
ROC, calibration, Projector, Grad-CAM, confusion matrices, and final system
metrics.
