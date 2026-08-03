import os
import random
import datetime
import time
import argparse

import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from decoder.model import COMPRESSDecoder
from decoder.data import RandomKPerMoleculeDataset, make_training_batch, move_batch_to_device
from decoder.losses import compute_loss
from decoder.utils import save_checkpoint, load_latest_checkpoint, init_loss_log, write_loss_log


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the COMPRESS decoder")
    p.add_argument("--config", default="configs/decoder.yaml")
    cli_args = p.parse_args()
    cfg = load_config(cli_args.config)

    SEED = cfg["seed"]
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=2))
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        world_size = 1
        global_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = (global_rank == 0)

    paths = cfg["paths"]
    train_cfg = cfg["training"]
    vocab_cfg = cfg["vocab"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss_weights"]

    input_file = paths["input_file"]
    checkpoint_dir = paths["checkpoint_dir"]
    log_file = paths["log_file"]

    path_noise = torch.tensor(train_cfg["path_noise"], dtype=torch.float32, device="cpu")
    atom_mask_id = vocab_cfg["atom_mask_id"]
    bond_mask_id = vocab_cfg["bond_mask_id"]

    if is_main:
        init_loss_log(log_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Data file '{input_file}' not found in {os.getcwd()}.")

    raw_dataset = torch.load(input_file, map_location="cpu")
    if is_main:
        print(f"Loaded raw molecules (train): {len(raw_dataset)}")

    train_raw = raw_dataset
    dataset = RandomKPerMoleculeDataset(
        raw_dataset=train_raw,
        k_per_molecule_per_epoch=train_cfg["k_per_molecule_per_epoch"],
        restrict_k_gt=0,
    )
    if is_main:
        print(f"Initial training items per epoch: {len(dataset)}")

    def collate_fn(samples):
        return make_training_batch(
            samples=samples, device="cpu",
            path_noise=path_noise,
            atom_mask_id=atom_mask_id, bond_mask_id=bond_mask_id,
        )

    batch_size = train_cfg["batch_size"]
    if ddp:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, collate_fn=collate_fn, drop_last=True, pin_memory=True,
            persistent_workers=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn, drop_last=False,
        )

    model = COMPRESSDecoder(
        hidden_dim=model_cfg["hidden_dim"], k_hidden_dim=model_cfg["k_hidden_dim"],
        edge_dim=model_cfg["edge_dim"], time_dim=model_cfg["time_dim"],
        atom_emb_dim=model_cfg["atom_emb_dim"], bond_emb_dim=model_cfg["bond_emb_dim"],
        n_rbf=model_cfg["n_rbf"], rbf_dmax=model_cfg["rbf_dmax"],
        n_blocks=model_cfg["n_blocks"], k_blocks=model_cfg["k_blocks"],
        num_atom_types=vocab_cfg["num_atom_types"], num_bond_types=vocab_cfg["num_bond_types"],
        num_atom_classes=vocab_cfg["num_atom_classes"], num_bond_classes=vocab_cfg["num_bond_classes"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )

    start_epoch = load_latest_checkpoint(
        checkpoint_dir=checkpoint_dir, model=model, optimizer=optimizer, device=device,
    )

    log_every_batches = train_cfg["log_every_batches"]
    grad_clip = train_cfg["grad_clip"]
    num_epochs = train_cfg["num_epochs"]

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    for epoch in range(start_epoch, num_epochs + 1):
        random.seed(1000 + epoch)
        dataset.resample_index()
        if ddp:
            sampler.set_epoch(epoch)
        model.train()

        epoch_start = time.time()
        ckpt_start = epoch_start
        total_loss = total_pos = total_charge = total_atom = total_bond = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            outputs = model(
                St=batch["St"], At=batch["At"], Bt=batch["Bt"], t=batch["t"],
                V=batch["V"], atom_valid=batch["atom_valid"], site_valid=batch["site_valid"],
            )
            loss, loss_dict = compute_loss(
                outputs=outputs, batch=batch,
                lambda_pos=loss_cfg["lambda_pos"], lambda_charge=loss_cfg["lambda_charge"],
                lambda_atom=loss_cfg["lambda_atom"], lambda_bond=loss_cfg["lambda_bond"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_pos += float(loss_dict["loss_pos"].cpu())
            total_charge += float(loss_dict["loss_charge"].cpu())
            total_atom += float(loss_dict["loss_atom"].cpu())
            total_bond += float(loss_dict["loss_bond"].cpu())
            num_batches += 1

            if (batch_idx + 1) % log_every_batches == 0 and is_main:
                now = time.time()
                since_last = now - ckpt_start
                elapsed = now - epoch_start
                done = batch_idx + 1
                total_batches = len(loader)
                rate = done / max(elapsed, 1e-9)
                remaining = (total_batches - done) / max(rate, 1e-9)
                ckpt_start = now
                running_loss = total_loss / max(num_batches, 1)
                print(f"  [epoch {epoch:04d} batch {done}/{total_batches}] "
                      f"loss={running_loss:.4f} | "
                      f"last {log_every_batches} batches: {since_last:.0f}s "
                      f"({since_last/log_every_batches*1000:.0f}ms/batch) | "
                      f"epoch elapsed {elapsed/60:.1f}min | "
                      f"epoch ETA {remaining/60:.1f}min")

        avg_loss = total_loss / max(num_batches, 1)
        avg_pos = total_pos / max(num_batches, 1)
        avg_charge = total_charge / max(num_batches, 1)
        avg_atom = total_atom / max(num_batches, 1)
        avg_bond = total_bond / max(num_batches, 1)
        epoch_loss_dict = {
            "loss": avg_loss, "loss_pos": avg_pos, "loss_charge": avg_charge,
            "loss_atom": avg_atom, "loss_bond": avg_bond,
        }

        if is_main:
            epoch_time = time.time() - epoch_start
            print(
                f"[Epoch {epoch:04d}] avg_loss={avg_loss:.4f} | pos={avg_pos:.4f} | chg={avg_charge:.4f} | "
                f"atom={avg_atom:.4f} | bond={avg_bond:.4f} | "
                f"epoch_time={epoch_time/60:.1f}min ({epoch_time/3600:.2f}h)"
            )
            write_loss_log(log_file=log_file, epoch=epoch, train=epoch_loss_dict)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir, epoch=epoch, model=model,
                optimizer=optimizer, loss_dict=epoch_loss_dict, config=cfg,
            )

        if ddp:
            dist.barrier()

    if is_main:
        print("Training finished.")
    if ddp:
        dist.destroy_process_group()
