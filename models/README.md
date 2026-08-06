# Model artifacts

Colab model files persist in the Drive project:

- `best/colab/` contains validation-selected models.
- `checkpoints/colab/` contains resumable checkpoints.

The Kaggle notebook writes its models directly under
`/kaggle/working/ugvnet/models/` inside Kaggle, so no repository-side Kaggle
folder is needed.

Weight files (`.pt`, `.pth`, and `.onnx`) are ignored by Git.
