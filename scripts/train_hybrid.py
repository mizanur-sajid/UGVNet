"""Train the recommended EfficientNetV2-S + ConvNeXt-Tiny UGVNet.

UGVNet strongly recommends a tuned, leakage-checked dataset with this layout:

    dataset/
      train/<class_name>/*
      validation/<class_name>/*
      test/<class_name>/*

The validation set is used for model selection. The test set is evaluated once,
after training, using the best validation checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from ugvnet.data_audit import audit_dataset, enforce_audit_policy
from ugvnet.hybrid import UGVNetHybrid, ugvnet_hybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2-S + ConvNeXt-Tiny UGVNet"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fusion-learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fusion-channels", type=int, default=384)
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--training-mode",
        choices=("auto", "small", "large"),
        default="auto",
        help="Auto uses small mode below 20,000 training images.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=None,
        help="Default: 5 in small mode and 0 in large mode.",
    )
    parser.add_argument(
        "--class-balance",
        choices=("auto", "on", "off"),
        default="auto",
        help="Auto enables balanced sampling when max/min class count >= 1.5.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write performance-aware TensorBoard diagnostics.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path("tensorboard"),
        help="Root directory for timestamped TensorBoard runs.",
    )
    parser.add_argument(
        "--tensorboard-histogram-interval",
        type=int,
        default=5,
        help="Log selected parameters and gradients every N epochs; 0 disables.",
    )
    parser.add_argument(
        "--tensorboard-profile-batches",
        type=int,
        default=5,
        help="Profile this many batches in epoch one; 0 disables.",
    )
    parser.add_argument(
        "--audit-policy",
        choices=("strict", "warn", "off"),
        default="strict",
        help="Scan every image before training; strict stops on critical issues.",
    )
    parser.add_argument(
        "--audit-workers",
        type=int,
        default=0,
        help="Audit worker threads; zero selects automatically.",
    )
    parser.add_argument("--run-name", default="ugvnet_hybrid")

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable PyTorch 2.0 model compilation (may cause OOM on 16GB T4 GPUs).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_dataset_layout(data_dir: Path) -> dict[str, Path]:
    """Require independent train, validation, and test folders."""
    split_paths = {split: data_dir / split for split in ("train", "validation", "test")}
    missing = [name for name, path in split_paths.items() if not path.is_dir()]
    if missing:
        expected = "\n".join(
            f"  {data_dir / split}/<class_name>/"
            for split in ("train", "validation", "test")
        )
        raise FileNotFoundError(
            "UGVNet strongly recommends a tuned dataset with independent "
            "train, validation, and test folders.\n"
            f"Missing folders: {', '.join(missing)}\nExpected:\n{expected}"
        )
    return split_paths


def build_transforms(
    image_size: int, training_mode: str
) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    crop_scale = (0.60, 1.0) if training_mode == "small" else (0.75, 1.0)
    augmentation_magnitude = 7 if training_mode == "small" else 5
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=crop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandAugment(num_ops=2, magnitude=augmentation_magnitude),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.20 if training_mode == "small" else 0.10,
                scale=(0.02, 0.12),
            ),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def resolve_training_mode(requested: str, training_examples: int) -> str:
    if requested != "auto":
        return requested
    return "small" if training_examples < 20_000 else "large"


def class_counts(dataset: datasets.ImageFolder) -> Counter[int]:
    return Counter(int(target) for target in dataset.targets)


def build_balanced_sampler(
    dataset: datasets.ImageFolder, setting: str
) -> WeightedRandomSampler | None:
    counts = class_counts(dataset)
    imbalance_ratio = max(counts.values()) / min(counts.values())
    enabled = setting == "on" or (setting == "auto" and imbalance_ratio >= 1.5)
    if not enabled:
        return None
    sample_weights = [1.0 / counts[int(target)] for target in dataset.targets]
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_datasets(
    data_dir: Path, image_size: int, requested_mode: str
) -> tuple[dict[str, datasets.ImageFolder], str]:
    split_paths = validate_dataset_layout(data_dir)
    provisional = datasets.ImageFolder(split_paths["train"])
    training_mode = resolve_training_mode(requested_mode, len(provisional))
    train_transform, evaluation_transform = build_transforms(image_size, training_mode)
    loaded = {
        "train": datasets.ImageFolder(split_paths["train"], train_transform),
        "validation": datasets.ImageFolder(
            split_paths["validation"], evaluation_transform
        ),
        "test": datasets.ImageFolder(split_paths["test"], evaluation_transform),
    }
    reference_classes = loaded["train"].classes
    for split_name in ("validation", "test"):
        if loaded[split_name].classes != reference_classes:
            raise ValueError(
                f"{split_name} class folders must exactly match train class folders."
            )
    return loaded, training_mode


def build_loaders(
    loaded: dict[str, datasets.ImageFolder],
    batch_size: int,
    num_workers: int,
    balance_setting: str,
    use_cuda: bool,
) -> tuple[dict[str, DataLoader[Any]], bool]:
    sampler = build_balanced_sampler(loaded["train"], balance_setting)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0,
    }
    loaders = {
        "train": DataLoader(
            loaded["train"],
            shuffle=sampler is None,
            sampler=sampler,
            **common,
        ),
        "validation": DataLoader(loaded["validation"], shuffle=False, **common),
        "test": DataLoader(loaded["test"], shuffle=False, **common),
    }
    return loaders, sampler is not None


def macro_f1_from_confusion(confusion: Tensor) -> float:
    confusion = confusion.float()
    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    per_class = torch.where(
        denominator > 0,
        2 * true_positive / denominator,
        torch.zeros_like(denominator),
    )
    return per_class.mean().item()


def selected_named_parameters(
    model: nn.Module, limit: int = 20
) -> list[tuple[str, nn.Parameter]]:
    """Pick informative tensors without dumping both complete backbones."""
    preferred = (
        "fusion.",
        "global_fusion.",
        "final_norm.",
        "head.",
        "efficientnet_features.7",
        "convnext_features.7",
    )
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(preferred)
    ]
    if len(selected) < limit:
        chosen = {name for name, _ in selected}
        selected.extend(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name not in chosen
        )
    return selected[:limit]


def log_selected_histograms(
    writer: SummaryWriter,
    model: nn.Module,
    step: int,
    *,
    gradients: bool,
) -> None:
    prefix = "Gradients" if gradients else "Parameters"
    for name, parameter in selected_named_parameters(model):
        values = parameter.grad if gradients else parameter.detach()
        if values is None or values.numel() == 0:
            continue
        writer.add_histogram(f"{prefix}/{name}", values.detach().float().cpu(), step)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_clip: float = 0.0,
    writer: SummaryWriter | None = None,
    tensorboard_step: int = 0,
    collect_outputs: bool = False,
    collect_fusion_diagnostics: bool = False,
    log_gradient_histograms: bool = False,
    profile_batches: int = 0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    gradient_norm = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    gate_sum = torch.zeros(2, dtype=torch.float64)
    gate_entropy_sum = 0.0
    gate_dominance_sum = 0.0
    efficientnet_selection_sum = 0.0
    fused_norm_sum = 0.0
    diagnostic_examples = 0
    all_probabilities: list[Tensor] = []
    all_targets: list[Tensor] = []
    all_embeddings: list[Tensor] = []
    gradient_context = torch.enable_grad if training else torch.no_grad

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    profiling = writer is not None and profile_batches > 0
    profiler_context = (
        profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=1, active=max(profile_batches - 1, 1), repeat=1),
            on_trace_ready=tensorboard_trace_handler(str(writer.log_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            acc_events=True,
        )
        if profiling
        else nullcontext()
    )

    with profiler_context as active_profiler, gradient_context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                fusion_weights: Tensor | None = None
                fused_features: Tensor | None = None
                if collect_outputs or collect_fusion_diagnostics:
                    if not isinstance(model, UGVNetHybrid):
                        raise TypeError("Fusion diagnostics require UGVNetHybrid.")
                    fused_features, fusion_weights = model.forward_features(
                        images, return_fusion_weights=True
                    )
                    logits = model.forward_head(fused_features)
                else:
                    logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is None:
                    raise RuntimeError("Training requires a gradient scaler.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    gradient_norm = float(
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    )
                if log_gradient_histograms and writer is not None:
                    log_selected_histograms(
                        writer, model, tensorboard_step, gradients=True
                    )
                scaler.step(optimizer)
                scaler.update()

            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == targets).sum().item()
            total_examples += batch_size
            encoded = targets.detach().cpu() * num_classes + predictions.detach().cpu()
            confusion += torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

            if fusion_weights is not None and fused_features is not None:
                weights = fusion_weights.detach().float()
                spatial_mean = weights.mean(dim=(2, 3))
                gate_sum += spatial_mean.sum(dim=0).cpu().double()
                entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
                gate_entropy_sum += entropy.mean(dim=(1, 2)).sum().item()
                gate_dominance_sum += weights.max(dim=1).values.mean(dim=(1, 2)).sum().item()
                efficientnet_selection_sum += (spatial_mean[:, 0] > spatial_mean[:, 1]).sum().item()
                embeddings = model.pool(fused_features).flatten(1)
                fused_norm_sum += embeddings.norm(dim=1).sum().item()
                diagnostic_examples += batch_size
                if collect_outputs:
                    all_probabilities.append(logits.softmax(dim=1).detach().cpu())
                    all_targets.append(targets.detach().cpu())
                    all_embeddings.append(embeddings.detach().cpu())

            if profiling:
                active_profiler.step()

    if total_examples == 0:
        raise ValueError("A dataset split contains no images.")
    result: dict[str, Any] = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "macro_f1": macro_f1_from_confusion(confusion),
        "gradient_norm": gradient_norm,
    }
    if collect_outputs:
        result["confusion"] = confusion
    if diagnostic_examples:
        result["fusion"] = {
            "efficientnet_gate_weight": gate_sum[0].item() / diagnostic_examples,
            "convnext_gate_weight": gate_sum[1].item() / diagnostic_examples,
            "gate_entropy": gate_entropy_sum / diagnostic_examples,
            "gate_dominance": gate_dominance_sum / diagnostic_examples,
            "efficientnet_selection_rate": efficientnet_selection_sum / diagnostic_examples,
            "fused_embedding_norm": fused_norm_sum / diagnostic_examples,
        }
    if collect_outputs:
        result["_tensorboard_outputs"] = {
            "probabilities": torch.cat(all_probabilities),
            "targets": torch.cat(all_targets),
            "embeddings": torch.cat(all_embeddings),
        }
    return result


def optimizer_for(model: UGVNetHybrid, args: argparse.Namespace) -> AdamW:
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": list(model.backbone_parameters()),
            "lr": args.backbone_learning_rate,
        },
        {
            "params": list(model.new_parameters()),
            "lr": args.fusion_learning_rate,
        },
    ]
    return AdamW(parameter_groups, weight_decay=args.weight_decay)


def save_checkpoint(
    path: Path,
    model: UGVNetHybrid,
    optimizer: AdamW,
    epoch: int,
    classes: list[str],
    validation_metrics: dict[str, float],
    args: argparse.Namespace,
    training_mode: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "classes": classes,
            "validation_metrics": validation_metrics,
            "training_mode": training_mode,
            "model_config": {
                "num_classes": len(classes),
                "pretrained": False,
                "fusion_channels": args.fusion_channels,
                "attention_heads": args.attention_heads,
                "fusion_depth": args.fusion_depth,
                "dropout": args.dropout,
            },
        },
        path,
    )


def format_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    result["data_dir"] = str(result["data_dir"])
    result["results_dir"] = str(result["results_dir"])
    result["models_dir"] = str(result["models_dir"])
    result["tensorboard_dir"] = str(result["tensorboard_dir"])
    return result


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)
    device = resolve_device(args.device)
    results_dir = args.results_dir / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_report: dict[str, Any] | None = None
    if args.audit_policy != "off":
        cached_audit_path = args.data_dir / "dataset_audit.json"
        if cached_audit_path.exists():
            print(f"Loading cached dataset audit from {cached_audit_path}")
            audit_report = json.loads(cached_audit_path.read_text(encoding="utf-8"))
            enforce_audit_policy(audit_report, args.audit_policy)
            import shutil
            shutil.copy2(cached_audit_path, results_dir / "dataset_audit.json")
        else:
            audit_report = audit_dataset(
                args.data_dir,
                report_path=cached_audit_path,
                workers=args.audit_workers or None,
            )
            enforce_audit_policy(audit_report, args.audit_policy)
            import shutil
            shutil.copy2(cached_audit_path, results_dir / "dataset_audit.json")
    else:
        print("WARNING: full-dataset audit is disabled.")
    loaded, training_mode = build_datasets(
        args.data_dir, args.image_size, args.training_mode
    )
    loaders, balanced_sampling = build_loaders(
        loaded,
        args.batch_size,
        args.num_workers,
        args.class_balance,
        device.type == "cuda",
    )
    classes = loaded["train"].classes
    freeze_epochs = args.freeze_backbone_epochs
    if freeze_epochs is None:
        freeze_epochs = 5 if training_mode == "small" else 0
    if freeze_epochs < 0:
        raise ValueError("freeze-backbone-epochs must be non-negative.")

    model = ugvnet_hybrid(
        num_classes=len(classes),
        pretrained=args.pretrained,
        fusion_channels=args.fusion_channels,
        attention_heads=args.attention_heads,
        fusion_depth=args.fusion_depth,
        dropout=args.dropout,
    ).to(device)
    if freeze_epochs > 0:
        model.set_backbones_trainable(False)

    base_model = model
    if args.compile:
        print("Compiling model for faster training...")
        model = torch.compile(model)

    optimizer = optimizer_for(base_model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_models_dir = args.models_dir / "best"
    checkpoints_dir = args.models_dir / "checkpoints"
    for directory in (results_dir, best_models_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (results_dir / "classes.json").write_text(
        json.dumps(classes, indent=2), encoding="utf-8"
    )
    (results_dir / "training_config.json").write_text(
        json.dumps(serializable_args(args), indent=2), encoding="utf-8"
    )

    writer: SummaryWriter | None = None
    tensorboard_run_dir: Path | None = None
    if args.tensorboard:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        tensorboard_run_dir = (
            args.tensorboard_dir / f"{args.run_name}_{timestamp}"
        )
        writer = SummaryWriter(
            log_dir=str(tensorboard_run_dir),
            max_queue=20,
            flush_secs=30,
        )
        writer.add_text(
            "Run/configuration",
            "```json\n"
            + json.dumps(serializable_args(args), indent=2)
            + "\n```",
        )
        writer.add_text("Run/classes", ", ".join(classes))
        writer.add_scalar(
            "Model/total_parameters",
            sum(parameter.numel() for parameter in base_model.parameters()),
            0,
        )
        if audit_report is not None:
            for key, value in audit_report["summary"].items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Audit/{key}", value, 0)
            writer.add_scalar(
                "Audit/duration_seconds",
                audit_report["duration_seconds"],
                0,
            )
        was_training = model.training
        example: Tensor | None = None
        try:
            model.eval()
            example = torch.zeros(
                1, 3, args.image_size, args.image_size, device=device
            )
            writer.add_graph(model, example)
        except Exception as error:  # noqa: BLE001
            writer.add_text("Warnings/model_graph", str(error), 0)
            print(f"TensorBoard graph could not be recorded: {error}")
        finally:
            model.train(was_training)
            del example
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log_selected_histograms(writer, base_model, 0, gradients=False)
        writer.flush()

    print(
        f"UGVNet hybrid | mode={training_mode} | device={device} | "
        f"train={len(loaded['train'])} | validation={len(loaded['validation'])} | "
        f"test={len(loaded['test'])} | classes={len(classes)} | "
        f"balanced_sampling={balanced_sampling} | freeze_epochs={freeze_epochs}"
    )
    best_f1 = -1.0
    epochs_without_improvement = 0
    best_path = best_models_dir / f"{args.run_name}_best.pt"
    last_path = checkpoints_dir / f"{args.run_name}_last.pt"
    print(
        f"Results: {results_dir} | Best model: {best_path} | "
        f"Checkpoints: {checkpoints_dir}"
    )

    for epoch in range(1, args.epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            base_model.set_backbones_trainable(True)
            print("Unfroze EfficientNetV2-S and ConvNeXt-Tiny backbones.")

        histogram_due = (
            writer is not None
            and args.tensorboard_histogram_interval > 0
            and epoch % args.tensorboard_histogram_interval == 0
        )
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            len(classes),
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip=args.gradient_clip,
            writer=writer,
            tensorboard_step=epoch,
            log_gradient_histograms=histogram_due,
            profile_batches=(
                args.tensorboard_profile_batches if epoch == 1 else 0
            ),
        )
        validation_metrics = run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            len(classes),
            amp_enabled=amp_enabled,
            collect_fusion_diagnostics=writer is not None,
        )
        if writer is not None:
            for split_name, metrics in (
                ("train", train_metrics),
                ("validation", validation_metrics),
            ):
                for metric_name in ("loss", "accuracy", "macro_f1"):
                    writer.add_scalar(
                        f"Epoch/{split_name}/{metric_name}",
                        metrics[metric_name],
                        epoch,
                    )
            writer.add_scalar(
                "Epoch/train/gradient_norm", train_metrics["gradient_norm"], epoch
            )
            writer.add_scalar(
                "Generalization/accuracy_gap",
                train_metrics["accuracy"] - validation_metrics["accuracy"],
                epoch,
            )
            for metric_name, value in validation_metrics.get("fusion", {}).items():
                writer.add_scalar(f"Hybrid/{metric_name}", value, epoch)
            if histogram_due:
                log_selected_histograms(writer, model, epoch, gradients=False)
            for group_index, parameter_group in enumerate(optimizer.param_groups):
                writer.add_scalar(
                    f"Learning_rate/group_{group_index}",
                    parameter_group["lr"],
                    epoch,
                )
        scheduler.step()
        print(
            f"epoch={epoch:03d} train[{format_metrics(train_metrics)}] "
            f"validation[{format_metrics(validation_metrics)}]"
        )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            classes,
            validation_metrics,
            args,
            training_mode,
        )

        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                classes,
                validation_metrics,
                args,
                training_mode,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        len(classes),
        amp_enabled=amp_enabled,
        collect_outputs=writer is not None,
        collect_fusion_diagnostics=writer is not None,
    )
    tensorboard_outputs = test_metrics.pop("_tensorboard_outputs", None)
    test_confusion = test_metrics.pop("confusion")
    test_metrics.pop("gradient_norm", None)
    final_metrics = {
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation_metrics"],
        "test": test_metrics,
        "tensorboard_run_dir": (
            str(tensorboard_run_dir)
            if tensorboard_run_dir is not None
            else None
        ),
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )
    print(f"Final test [{format_metrics(test_metrics)}]")
    if writer is not None:
        for metric_name in ("loss", "accuracy", "macro_f1"):
            writer.add_scalar(
                f"Test/{metric_name}", test_metrics[metric_name], checkpoint["epoch"]
            )
        for metric_name, value in test_metrics.get("fusion", {}).items():
            writer.add_scalar(f"Hybrid/test_{metric_name}", value, checkpoint["epoch"])
        confusion_image = test_confusion.float()
        confusion_image /= confusion_image.max().clamp_min(1)
        writer.add_image(
            "Evaluation/confusion_matrix",
            confusion_image.unsqueeze(0),
            checkpoint["epoch"],
        )
        if tensorboard_outputs is not None:
            probabilities = tensorboard_outputs["probabilities"]
            targets = tensorboard_outputs["targets"]
            for class_index, class_name in enumerate(classes):
                writer.add_pr_curve(
                    f"PR/{class_name}",
                    (targets == class_index).int(),
                    probabilities[:, class_index],
                    checkpoint["epoch"],
                )
            writer.add_embedding(
                tensorboard_outputs["embeddings"],
                label_img=tensorboard_outputs.get("sample_images"),
                metadata=[classes[index] for index in targets.tolist()],
                tag="Test/fused_embeddings",
                global_step=checkpoint["epoch"],
            )
            if tensorboard_outputs.get("sample_images") is not None:
                writer.add_images("Test/Sample_Predictions", tensorboard_outputs["sample_images"], checkpoint["epoch"])

        hparams = {
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "fusion_lr": args.fusion_learning_rate,
            "backbone_lr": args.backbone_learning_rate,
            "fusion_channels": args.fusion_channels,
            "fusion_depth": args.fusion_depth,
            "dropout": args.dropout,
            "training_mode": training_mode,
        }
        writer.add_hparams(
            hparams,
            {
                "hparam/validation_accuracy": checkpoint["validation_metrics"]["accuracy"],
                "hparam/validation_macro_f1": checkpoint["validation_metrics"]["macro_f1"],
                "hparam/test_accuracy": test_metrics["accuracy"],
                "hparam/test_macro_f1": test_metrics["macro_f1"],
            },
            run_name="hparams",
        )
        writer.add_text(
            "Final/summary",
            f"Best epoch: {checkpoint['epoch']}  \n"
            f"Test accuracy: {test_metrics['accuracy']:.2%}  \n"
            f"Test macro-F1: {test_metrics['macro_f1']:.4f}",
            checkpoint["epoch"],
        )
        writer.flush()
        writer.close()
        print(f"TensorBoard run: {tensorboard_run_dir}")


if __name__ == "__main__":
    main()
