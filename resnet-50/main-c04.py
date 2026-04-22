import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
from datetime import datetime
from tqdm import tqdm

# -----------------------------
# CONFIG - c04-s140-w10
# -----------------------------
DATA_DIR = "swin-images-c04-w10-1536"
BATCH_SIZE = 4
NUM_CLASSES = 4
LR = 1e-4
EPOCHS = 100
USE_AMP = True
IMG_SIZE = 1536
NUM_WORKERS = 16
CHECKPOINT_DIR = "out/c04-w10-1536/"
LOG_PATH = "out/c04-w10-1536/log_rank0.txt"

# -----------------------------
# CONFIG - c10-s84-w10
# -----------------------------
#DATA_DIR = "swin-images-c10-w10-1536-3"
#BATCH_SIZE = 4
#NUM_CLASSES = 4
#LR = 1e-4
#EPOCHS = 100
#USE_AMP = True
#IMG_SIZE = 1536
#NUM_WORKERS = 16
#CHECKPOINT_DIR = "out/c10-w10-1536/"
#LOG_PATH = "out/c10-w10-1536/log_rank0.txt"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Parse CLI args
EVAL_ONLY = any(arg.lower() == "eval=true" for arg in sys.argv)
SELECTED_CKPT = None
for arg in sys.argv:
    if arg.startswith("ckpt="):
        SELECTED_CKPT = arg.split("=")[1]

# -----------------------------
# LOGGING
# -----------------------------
def log_rank0(msg="", local_rank=0):
    """Prints to console AND appends to log file (rank 0 only)."""
    if local_rank == 0:
        # Print to stdout
        print(msg)
        sys.stdout.flush()

        # Append to log file
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

# -----------------------------
# DDP SETUP
# -----------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()


# -----------------------------
# CHECKPOINT HELPERS
# -----------------------------
def find_latest_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir)
             if f.startswith("epoch_") and f.endswith(".pt")]
    if len(ckpts) == 0:
        return None
    epochs = [int(f.split("_")[1].split(".")[0]) for f in ckpts]
    latest_epoch = max(epochs)
    latest_file = f"epoch_{latest_epoch}.pt"
    return os.path.join(checkpoint_dir, latest_file), latest_epoch


def save_checkpoint(epoch, model, optimizer, scaler, path):
    state = {
        "epoch": epoch,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"]


# -----------------------------
# MAIN
# -----------------------------
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # -----------------------------
    # TRANSFORMS
    # -----------------------------
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -----------------------------
    # DATASETS
    # -----------------------------
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)

    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # -----------------------------
    # MODEL
    # -----------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # -----------------------------
    # EVAL-ONLY MODE
    # -----------------------------
    if EVAL_ONLY:
        if local_rank == 0:
            log_rank0("Running in EVAL-ONLY mode")

        # If user selected a checkpoint, use it
        if SELECTED_CKPT is not None:
            ckpt_path = os.path.join(CHECKPOINT_DIR, SELECTED_CKPT)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Requested checkpoint not found: {ckpt_path}")
            if local_rank == 0:
                log_rank0(f"Loading selected checkpoint: {ckpt_path}")
        else:
            # Otherwise load latest
            latest = find_latest_checkpoint(CHECKPOINT_DIR)
            if latest is None:
                raise FileNotFoundError("No checkpoints found for eval mode.")
            ckpt_path, _ = latest
            if local_rank == 0:
                log_rank0(f"Loading latest checkpoint: {ckpt_path}")

        load_checkpoint(ckpt_path, model, optimizer, scaler)

        # ---- VALIDATION ONLY ----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        # Metrics
        num_classes = 4
        conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

        if local_rank == 0:
            pbar = tqdm(total=len(val_loader), desc="Eval [Val]")
        else:
            pbar = None

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

                # Update confusion matrix
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    conf_mat[t.long(), p.long()] += 1

                if local_rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(loss=val_loss / val_total,
                                    acc=100 * val_correct / val_total)

        if local_rank == 0:
            pbar.close()

            # ---- METRICS ----
            tp = conf_mat.diag()
            fp = conf_mat.sum(0) - tp
            fn = conf_mat.sum(1) - tp

            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)

            macro_precision = precision.mean().item()
            macro_recall = recall.mean().item()
            macro_f1 = f1.mean().item()

            log_rank0(f"\nEval Results:")
            log_rank0(f"  Val Loss: {val_loss/val_total:.4f}")
            log_rank0(f"  Val Acc:  {100*val_correct/val_total:.2f}%")
            log_rank0(f"  Precision: {macro_precision:.4f} | Recall: {macro_recall:.4f} | F1: {macro_f1:.4f}")

            # Per-class breakdown
            log_rank0("\n  Per-Class Metrics:")
            for c in range(num_classes):
                log_rank0(f"    Class {c}: Precision {precision[c]:.3f}  Recall {recall[c]:.3f}  F1 {f1[c]:.3f}")
            log_rank0()

        cleanup_ddp()
        return


    # -----------------------------
    # TRAINING MODE
    # -----------------------------
    latest = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        ckpt_path, last_epoch = latest
        if local_rank == 0:
            log_rank0(f"Resuming from checkpoint: {ckpt_path}")
        start_epoch = load_checkpoint(ckpt_path, model, optimizer, scaler) + 1
    else:
        start_epoch = 1

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(start_epoch, EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        # ---- TRAIN ----
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        if local_rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch} [Train]")
        else:
            pbar = None

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            if local_rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=running_loss / total,
                                 acc=100 * correct / total)

        if local_rank == 0:
            pbar.close()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        # For precision/recall/F1
        num_classes = 4
        conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

        if local_rank == 0:
            pbar = tqdm(total=len(val_loader), desc=f"Epoch {epoch} [Val]")
        else:
            pbar = None

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

                # Update confusion matrix
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    conf_mat[t.long(), p.long()] += 1

                if local_rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(loss=val_loss / val_total,
                                    acc=100 * val_correct / val_total)

        if local_rank == 0:
            pbar.close()

            # ---- SAVE CHECKPOINT FOR THIS EPOCH ----
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
            save_checkpoint(epoch, model, optimizer, scaler, ckpt_path)

            # ---- METRICS ----
            tp = conf_mat.diag()
            fp = conf_mat.sum(0) - tp
            fn = conf_mat.sum(1) - tp

            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)

            macro_precision = precision.mean().item()
            macro_recall = recall.mean().item()
            macro_f1 = f1.mean().item()

            log_rank0(f"\nEpoch {epoch}:")
            log_rank0(f"  Train Loss: {running_loss/total:.4f} | Train Acc: {100*correct/total:.2f}%")
            log_rank0(f"  Val Loss:   {val_loss/val_total:.4f} | Val Acc:   {100*val_correct/val_total:.2f}%")
            log_rank0(f"  Precision:  {macro_precision:.4f} | Recall: {macro_recall:.4f} | F1: {macro_f1:.4f}\n")

            # Per-class breakdown
            for c in range(num_classes):
                log_rank0(f"  Class {c}: Precision {precision[c]:.3f}  Recall {recall[c]:.3f}  F1 {f1[c]:.3f}")
            log_rank0()


    cleanup_ddp()


if __name__ == "__main__":
    main()