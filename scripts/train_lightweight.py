"""Train UGVNet on an ImageFolder dataset.

Expected layout:
    data/
      train/class_name/image.jpg
      validation/class_name/image.jpg
      test/class_name/image.jpg

The validation set is used for model selection. The test set is evaluated once,
after training, using the best validation checkpoint.
"""

from __future__ import annotations

import argparse
import json
import matplotlib.pyplot as plt
import typing
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from torchvision import datasets, transforms

from ugvnet import create_ugvnet
from ugvnet.data_audit import audit_dataset, enforce_audit_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UGVNet")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("tiny", "small", "base"), default="tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write performance-aware TensorBoard summaries.",
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
        help="Log selected parameter histograms every N epochs; 0 disables.",
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
    parser.add_argument("--run-name", default="ugvnet_lightweight")
    parser.add_argument("--device", default="auto")

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


def build_loaders(
    data_dir: Path, image_size: int, batch_size: int, num_workers: int
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], list[str]]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.ImageFolder(data_dir / "train", train_transform)
    validation_dataset = datasets.ImageFolder(
        data_dir / "validation", validation_transform
    )
    test_dataset = datasets.ImageFolder(
        data_dir / "test", validation_transform
    )
    if train_dataset.classes != validation_dataset.classes:
        raise ValueError("Train and validation class folders must match.")
    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Test class folders must match train class folders.")

    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, validation_loader, test_loader, train_dataset.classes


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.AdamW | None = None,
    collect_outputs: bool = False,
    active_profiler = None,
    writer = None,
    epoch: int = 0,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, typing.Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    output_probabilities = []
    output_targets = []
    output_embeddings = []
    sample_images = []
    
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=training and device.type == "cuda"):
                if collect_outputs:
                    features = model.forward_features(images)
                    embeddings = model.pool(features).flatten(1)
                    logits = model.head(model.dropout(embeddings))
                else:
                    logits = model(images)
                    embeddings = None
                    
                loss = criterion(logits, targets)
            
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if writer is not None and batch_index == len(loader):
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if writer is not None and batch_index == len(loader):
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                    optimizer.step()
                
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()
            total_examples += batch_size
            
            if collect_outputs:
                output_probabilities.append(logits.detach().float().softmax(dim=1).cpu())
                output_targets.append(targets.detach().cpu())
                if embeddings is not None:
                    output_embeddings.append(embeddings.detach().float().cpu())
                
                remaining = 100 - sum(tensor.size(0) for tensor in sample_images)
                if remaining > 0:
                    take = min(remaining, batch_size)
                    sample_images.append(images[:take].detach().float().cpu())
                    
            if active_profiler is not None:
                active_profiler.step()

    result = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
    
    if collect_outputs:
        result["_tensorboard_outputs"] = {
            "probabilities": torch.cat(output_probabilities),
            "targets": torch.cat(output_targets),
            "embeddings": torch.cat(output_embeddings),
            "sample_images": torch.cat(sample_images) if sample_images else None,
        }
        
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    classes: list[str],
    args: argparse.Namespace,
    validation_accuracy: float,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "classes": classes,
        "args": vars(args),
        "validation_accuracy": validation_accuracy,
    }
    torch.save(checkpoint, path)


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
    train_loader, validation_loader, test_loader, classes = build_loaders(
        args.data_dir, args.image_size, args.batch_size, args.num_workers
    )
    model = create_ugvnet(
        args.variant, num_classes=len(classes), dropout=args.dropout
    ).to(device)

    base_model = model
    if args.compile:
        print("Compiling model for faster training...")
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    optimizer = AdamW(
        base_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_models_dir = args.models_dir / "best"
    checkpoints_dir = args.models_dir / "checkpoints"
    for directory in (results_dir, best_models_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (results_dir / "classes.json").write_text(
        json.dumps(classes, indent=2), encoding="utf-8"
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
        configuration = vars(args).copy()
        for key, value in configuration.items():
            if isinstance(value, Path):
                configuration[key] = str(value)
        writer.add_text(
            "Run/configuration",
            "```json\n" + json.dumps(configuration, indent=2) + "\n```",
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
        for name, parameter in list(base_model.named_parameters())[-20:]:
            writer.add_histogram(
                f"Parameters/{name}", parameter.detach().float().cpu(), 0
            )
        writer.flush()

    best_path = best_models_dir / f"{args.run_name}_best.pt"
    last_path = checkpoints_dir / f"{args.run_name}_last.pt"

    best_accuracy = 0.0
    epoch = 0
    print(f"Training ugvnet_{args.variant} on {device} with {len(classes)} classes.")
    
    # Setup profiler
    profiler_context = None
    if writer is not None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        profiler_context = profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=tensorboard_trace_handler(str(tensorboard_run_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            acc_events=True,
        )
        profiler_context.start()
        
    for epoch in range(1, args.epochs + 1):

        train_result = run_epoch(
            model, train_loader, criterion, device, optimizer, writer=writer, epoch=epoch, active_profiler=profiler_context if isinstance(profiler_context, profile) else None, scaler=scaler
        )
        train_loss = train_result["loss"]
        train_accuracy = train_result["accuracy"]
        validation_result = run_epoch(
            model, validation_loader, criterion, device
        )
        validation_loss = validation_result["loss"]
        validation_accuracy = validation_result["accuracy"]
        if writer is not None:
            writer.add_scalar("Epoch/train/loss", train_loss, epoch)
            writer.add_scalar("Epoch/train/accuracy", train_accuracy, epoch)
            writer.add_scalar("Epoch/validation/loss", validation_loss, epoch)
            writer.add_scalar(
                "Epoch/validation/accuracy", validation_accuracy, epoch
            )
            writer.add_scalar(
                "Learning_rate/group_0", optimizer.param_groups[0]["lr"], epoch
            )
            if (
                args.tensorboard_histogram_interval > 0
                and epoch % args.tensorboard_histogram_interval == 0
            ):
                for name, parameter in list(base_model.named_parameters())[-20:]:
                    writer.add_histogram(
                        f"Parameters/{name}", parameter.detach().float().cpu(), epoch
                    )
        scheduler.step()
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            classes,
            args,
            validation_accuracy,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                classes,
                args,
                validation_accuracy,
            )

    if profiler_context is not None:
        profiler_context.stop()

    print(f"Best validation accuracy: {best_accuracy:.4f}")

    # Evaluate the test set using the best checkpoint.
    best_checkpoint = torch.load(
        best_path, map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["model"])
    test_result = run_epoch(
        model, test_loader, criterion, device, collect_outputs=True
    )
    test_loss = test_result["loss"]
    test_accuracy = test_result["accuracy"]
    
    print(
        f"Final test  loss={test_loss:.4f} accuracy={test_accuracy:.4f}"
    )
    final_metrics = {
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )

    if writer is not None:
        writer.add_scalar("Best/validation_accuracy", best_accuracy, epoch)
        writer.add_scalar("Test/loss", test_loss, best_checkpoint["epoch"])
        writer.add_scalar(
            "Test/accuracy", test_accuracy, best_checkpoint["epoch"]
        )
        writer.add_text(
            "Final/summary",
            f"Best validation accuracy: {best_accuracy:.2%}  \n"
            f"Test accuracy: {test_accuracy:.2%}",
            epoch,
        )
        
        # Add 100% capacity TensorBoard features for the test set
        test_outputs = test_result.get("_tensorboard_outputs")
        if test_outputs:
            probs = test_outputs["probabilities"]
            targets = test_outputs["targets"]
            images = test_outputs.get("sample_images")
            
            # PR Curves
            for class_index, class_name in enumerate(classes):
                writer.add_pr_curve(
                    f"PR/{class_name}",
                    (targets == class_index).int(),
                    probs[:, class_index],
                    best_checkpoint["epoch"],
                )
                
            # Sample Predictions Images
            if images is not None:
                writer.add_images("Test/Sample_Predictions", images, best_checkpoint["epoch"])
                
            # Embeddings with image thumbnails
            embeddings = test_outputs.get("embeddings")
            if embeddings is not None:
                writer.add_embedding(
                    embeddings,
                    metadata=[classes[i] for i in targets.tolist()],
                    label_img=images,
                    tag="Test/embeddings",
                    global_step=best_checkpoint["epoch"],
                )
                
            # Confusion Matrix
            preds = probs.argmax(dim=1)
            cm = torch.zeros(len(classes), len(classes), dtype=torch.int64)
            for t, p in zip(targets, preds):
                cm[t.item(), p.item()] += 1
            
            fig, ax = plt.subplots(figsize=(10, 10))
            cax = ax.matshow(cm.numpy(), cmap="Blues")
            fig.colorbar(cax)
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=90)
            ax.set_yticklabels(classes)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, str(cm[i, j].item()), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
            writer.add_figure("Test/Confusion_Matrix", fig, best_checkpoint["epoch"])
            plt.close(fig)
        writer.add_hparams(
            {
                "variant": args.variant,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "dropout": args.dropout,
            },
            {
                "hparam/best_validation_accuracy": best_accuracy,
                "hparam/test_accuracy": test_accuracy,
            },
            run_name="hparams",
        )
        writer.flush()
        writer.close()
        print(f"TensorBoard run: {tensorboard_run_dir}")


if __name__ == "__main__":
    main()
