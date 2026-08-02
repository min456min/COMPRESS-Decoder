import torch
import torch.nn.functional as F


def masked_mse_loss(pred, target, mask):
    mask = mask.unsqueeze(-1).float()
    loss = (pred - target) ** 2
    loss = loss * mask
    denom = mask.sum() * pred.shape[-1]
    denom = denom.clamp(min=1.0)
    return loss.sum() / denom


def atom_ce_loss(atom_logits, atom_target, atom_valid, atom_is_masked):
    """CE on atom types, applied only to atoms still MASK at time t."""
    target = atom_target.clone()
    train_mask = atom_valid & atom_is_masked
    target = target.masked_fill(~train_mask, -100)
    B, M, C = atom_logits.shape
    loss = F.cross_entropy(
        atom_logits.reshape(B * M, C),
        target.reshape(B * M),
        ignore_index=-100,
    )
    return loss


def bond_ce_loss(bond_logits, bond_target, atom_valid, bond_is_masked):
    """Weighted CE on bond types, applied only to bonds still MASK at time t."""
    B, M, _, C = bond_logits.shape
    target = bond_target.clone()
    pair_valid = atom_valid[:, :, None] & atom_valid[:, None, :]
    eye = torch.eye(M, device=bond_logits.device, dtype=torch.bool).view(1, M, M)
    pair_valid = pair_valid & (~eye)
    train_mask = pair_valid & bond_is_masked
    target = target.masked_fill(~train_mask, -100)
    bond_weight = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], dtype=bond_logits.dtype, device=bond_logits.device)
    loss = F.cross_entropy(
        bond_logits.reshape(B * M * M, C),
        target.reshape(B * M * M),
        weight=bond_weight,
        ignore_index=-100,
    )
    return loss


def compute_loss(
    outputs,
    batch,
    lambda_pos=3.0,
    lambda_charge=1.0,
    lambda_atom=0.4,
    lambda_bond=2.0,
):
    S1_pred, atom_logits, bond_logits = outputs
    loss_pos = masked_mse_loss(
        pred=S1_pred[:, :, :3], target=batch["S1"][:, :, :3], mask=batch["atom_valid"])
    loss_charge = masked_mse_loss(
        pred=S1_pred[:, :, 3:4], target=batch["S1"][:, :, 3:4], mask=batch["atom_valid"])
    loss_atom = atom_ce_loss(
        atom_logits=atom_logits, atom_target=batch["A"],
        atom_valid=batch["atom_valid"], atom_is_masked=batch["atom_is_masked"],
    )
    loss_bond = bond_ce_loss(
        bond_logits=bond_logits, bond_target=batch["B"],
        atom_valid=batch["atom_valid"], bond_is_masked=batch["bond_is_masked"],
    )
    loss = (
        lambda_pos * loss_pos
        + lambda_charge * loss_charge
        + lambda_atom * loss_atom
        + lambda_bond * loss_bond
    )
    loss_dict = {
        "loss": loss.detach(),
        "loss_pos": loss_pos.detach(),
        "loss_charge": loss_charge.detach(),
        "loss_atom": loss_atom.detach(),
        "loss_bond": loss_bond.detach(),
    }
    return loss, loss_dict
