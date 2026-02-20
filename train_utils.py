from __future__ import annotations

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


@torch.no_grad()
def accuracy_topk(logits, targets, ks=(1, 5)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / targets.size(0)).item())
    return res


def train_one_epoch(model, loader, epoch, optimizer, cfg, device, criterion, scaler, lr_at_epoch=None):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0

    steps = len(loader)
    pbar = tqdm(loader, desc=f"Train {epoch+1}/{cfg.epochs}", leave=False)

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if lr_at_epoch is not None:
            epoch_float = epoch + step / max(steps, 1)
            lr = lr_at_epoch(epoch_float)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = optimizer.param_groups[0]["lr"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        top1, top5 = accuracy_topk(logits.detach(), targets, ks=(1, 5))
        running_loss += loss.item()
        running_top1 += top1
        running_top5 += top5

        pbar.set_postfix({
            "loss": f"{running_loss/(step+1):.4f}",
            "top1": f"{running_top1/(step+1):.4f}",
            "top5": f"{running_top5/(step+1):.4f}",
            "lr": f"{lr:.2e}",
        })

    return running_loss / steps, running_top1 / steps, running_top5 / steps


@torch.no_grad()
def evaluate(model, loader, epoch, cfg, device, criterion):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0

    pbar = tqdm(loader, desc=f"Val   {epoch+1}/{cfg.epochs}", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        top1, top5 = accuracy_topk(logits, targets, ks=(1, 5))
        running_loss += loss.item()
        running_top1 += top1
        running_top5 += top5

        pbar.set_postfix({
            "loss": f"{running_loss/(step+1):.4f}",
            "top1": f"{running_top1/(step+1):.4f}",
            "top5": f"{running_top5/(step+1):.4f}",
        })

    steps = len(loader)
    return running_loss / steps, running_top1 / steps, running_top5 / steps


def save_checkpoint(path, model, optimizer, epoch, best_metric, cfg, history=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg": cfg.__dict__,
        "history": history,
    }
    torch.save(ckpt, path)


def plot_history(history):
    if not isinstance(history, list) or len(history) == 0:
        print("history is empty; train first to plot curves.")
        return

    # show ticks at 2, 4, 6, ...
    epochs_range = range(1, len(history) + 1)
    epoch_ticks = list(range(2, len(epochs_range) + 1, 2))
    
    train_top1 = [h.get("train_top1") for h in history]
    val_top1 = [h.get("test_top1") for h in history]
    train_top5 = [h.get("train_top5") for h in history]
    val_top5 = [h.get("test_top5") for h in history]
    train_loss = [h.get("train_loss") for h in history]
    val_loss = [h.get("test_loss") for h in history]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_top1, label="Train Top1")
    plt.plot(epochs_range, val_top1, label="Test Top1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Top-1 Accuracy")
    plt.xticks(epoch_ticks)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_top5, label="Train Top5")
    plt.plot(epochs_range, val_top5, label="Test Top5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Top-5 Accuracy")
    plt.xticks(epoch_ticks)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.xticks(epoch_ticks)

    plt.tight_layout()
    plt.show()
