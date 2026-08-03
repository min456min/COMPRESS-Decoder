import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def _unwrap(model):
    """Return the underlying module whether or not model is DDP-wrapped."""
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, loss_dict, config):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_dict": loss_dict,
        "config": config,
    }
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    tmp_path = latest_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, latest_path)
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
    torch.save(checkpoint, epoch_path)
    print(f"Saved checkpoint: {epoch_path}")


def load_latest_checkpoint(checkpoint_dir, model, optimizer, device):
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if not os.path.exists(latest_path):
        print("No checkpoint found. Starting from epoch 1.")
        return 1
    checkpoint = torch.load(latest_path, map_location=device)
    _unwrap(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Loaded checkpoint: {latest_path}")
    print(f"Resuming from start of epoch {start_epoch}")
    return start_epoch


def init_loss_log(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir != "":
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,train_pos,train_charge,train_atom,train_bond\n")


def write_loss_log(log_file, epoch, train):
    with open(log_file, "a") as f:
        f.write(
            f"{epoch},"
            f"{train['loss']:.8f},{train['loss_pos']:.8f},{train['loss_charge']:.8f},"
            f"{train['loss_atom']:.8f},{train['loss_bond']:.8f}\n"
        )
