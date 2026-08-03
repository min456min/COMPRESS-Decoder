import math
import random
import torch
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment

# ============================================================
# Vocabulary
# ============================================================
# Atom input vocab (At):  0..10 real atom types, 11 = unused, 12 = MASK, num_atom_types = 13
# Atom output classes:    0..10 real, 11 = reserved/unused, num_atom_classes = 12
# Bond input vocab (Bt):  0 none, 1..4 real bond orders, 5 = MASK, num_bond_types = 6
# Bond output classes:    0..4, num_bond_classes = 5
# ============================================================
#
# Naming note: following the method section, V is the COMPRESS representation
# (K sites) and S is the all-atom structure (M atoms). "K" and "M" are still used for the scalar site-count / atom-count (as in the paper), e.g. K_value, K1/K2/... resolution keys, M_atoms.

def as_column(x):
    """Convert a 1D tensor from shape (N,) to (N, 1)."""
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x


def set_log(x, eps=1e-8):
    """Apply log with clamping to avoid log(0) / negative inputs."""
    return torch.log(torch.clamp(x, min=eps))


def make_S(S, device='cpu'):
    """Merge S (all-atom) features into (M, 4): [pos_x, pos_y, pos_z, chg]."""
    pos = S["pos"].float().to(device)
    chg = as_column(S["chg"].float().to(device))
    S = torch.cat([pos, chg], dim=-1)
    return S


def make_V(V, device='cpu'):
    """Merge V (COMPRESS) features into (K, 6): [pos_x, pos_y, pos_z, chg, log_sig, log_eps]."""
    pos = V["pos"].float().to(device)
    chg = as_column(V["chg"].float().to(device))
    sig = as_column(V["sig"].float().to(device))
    eps = as_column(V["eps"].float().to(device))
    V = torch.cat([pos, chg, set_log(sig), set_log(eps)], dim=-1)
    return V


def make_S0(
        S,
        V,
        device='cpu',
        min_dist=0.3,
        max_tries=1000,
        chg_noise_scale=0.3,
        pos_noise_scale=0.5,
        ):
    """Create an initial atom cloud S0 (4-dim: pos+chg) from COMPRESS sites V."""
    M_n = S.shape[0]
    K_n = V.shape[0]
    repeats = math.ceil(M_n / K_n)
    site_idx = torch.arange(K_n, device=device).repeat(repeats)[:M_n]
    site_idx = site_idx[torch.randperm(M_n, device=device)]
    V_pos = V[:, :3]
    V_sig = torch.exp(V[:, 4]).unsqueeze(-1).expand(-1, 3)
    S0_pos_list = []
    for i in range(M_n):
        k = site_idx[i]
        candidate = None
        accepted = False
        for _ in range(max_tries):
            candidate = V_pos[k] + pos_noise_scale * V_sig[k] * torch.randn(3, device=device)
            if len(S0_pos_list) == 0:
                accepted = True
                break
            prev = torch.stack(S0_pos_list, dim=0)
            d = torch.norm(prev - candidate.view(1, 3), dim=-1)
            if torch.all(d > min_dist):
                accepted = True
                break
        if not accepted:
            candidate = V_pos[k] + pos_noise_scale * V_sig[k] * torch.randn(3, device=device)
        S0_pos_list.append(candidate)
    S0_pos = torch.stack(S0_pos_list, dim=0)
    V_chg = V[:, 3]
    counts = torch.bincount(site_idx, minlength=K_n).float().clamp(min=1.0)
    S0_chg = V_chg[site_idx] / counts[site_idx]
    S0_chg = S0_chg.unsqueeze(-1)
    noise = chg_noise_scale * torch.randn(M_n, 1, device=device)
    for k in range(K_n):
        mask = site_idx == k
        if mask.any():
            noise[mask] = noise[mask] - noise[mask].mean(dim=0, keepdim=True)
    S0_chg = S0_chg + noise
    S0 = torch.cat([S0_pos, S0_chg], dim=-1)
    return S0


def hungarian_pair_pos_charge(target, query, w_pos=1.0, w_charge=0.1, normalize=True):
    """Match target atoms to query atoms using position + weak charge cost."""
    pos_cost = torch.cdist(query[:, :3], target[:, :3]).pow(2)
    charge_cost = (query[:, None, 3] - target[None, :, 3]).pow(2)
    if normalize:
        pos_cost = pos_cost / (pos_cost.detach().mean() + 1e-8)
        charge_cost = charge_cost / (charge_cost.detach().mean() + 1e-8)
    cost = w_pos * pos_cost + w_charge * charge_cost
    row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
    target_pair = target[col_idx]
    return target_pair, col_idx


def discrete_interpolant_atom(atom_target, t, atom_mask_id, device="cpu"):
    """Sample current atom state from a mask-to-data discrete path."""
    M = atom_target.shape[0]
    reveal_prob = float(t.detach().cpu().item())
    reveal = torch.rand(M, device=device) < reveal_prob
    atom_state = torch.full((M,), atom_mask_id, dtype=torch.long, device=device)
    atom_state[reveal] = atom_target[reveal]
    atom_loss_target = atom_target.clone()
    atom_is_masked = ~reveal
    return atom_state, atom_loss_target, atom_is_masked


def discrete_interpolant_bond(bond_target, t, bond_mask_id, device="cpu"):
    """Sample current bond state from a mask-to-data discrete path."""
    M = bond_target.shape[0]
    reveal_prob = float(t.detach().cpu().item())
    tri = torch.triu(torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1)
    reveal_tri = torch.zeros(M, M, device=device, dtype=torch.bool)
    reveal_tri[tri] = torch.rand(tri.sum(), device=device) < reveal_prob
    reveal = reveal_tri | reveal_tri.T
    bond_state = torch.full((M, M), bond_mask_id, dtype=torch.long, device=device)
    bond_state[reveal] = bond_target[reveal]
    idx = torch.arange(M, device=device)
    bond_state[idx, idx] = 0
    bond_loss_target = bond_target.clone()
    bond_loss_target[idx, idx] = -100
    bond_is_masked = bond_state == bond_mask_id
    bond_is_masked[idx, idx] = False
    return bond_state, bond_loss_target, bond_is_masked


def make_state(
    sample,
    device="cpu",
    path_noise=0.001,
    atom_mask_id=12,
    bond_mask_id=5,
    ):
    S_data = sample["AA"]
    V_data = sample["V"]
    S = make_S(S_data, device=device)
    V = make_V(V_data, device=device)
    atom_type = S_data["atom_type"].long().to(device)
    bond_type = S_data["bond_type"].long().to(device)
    S0 = make_S0(
        S, V, device=device,
        min_dist=0.3, max_tries=1000, chg_noise_scale=0.3, pos_noise_scale=0.5,
    )
    S_pair, col_idx = hungarian_pair_pos_charge(
        target=S, query=S0, w_pos=1.0, w_charge=0.05, normalize=True,
    )
    atom_target = atom_type[col_idx]
    bond_target = bond_type[col_idx][:, col_idx]
    t = torch.rand(1, device=device)   # t ~ U[0,1)
    St = (1.0 - (1.0 - path_noise) * t) * S0 + t * S_pair
    At, A, atom_is_masked = discrete_interpolant_atom(
        atom_target=atom_target, t=t, atom_mask_id=atom_mask_id, device=device,
    )
    Bt, B, bond_is_masked = discrete_interpolant_bond(
        bond_target=bond_target, t=t, bond_mask_id=bond_mask_id, device=device,
    )
    return {
        "St": St, "S0": S0, "S1": S_pair, "V": V, "t": t,
        "At": At, "A": A, "Bt": Bt, "B": B,
        "atom_is_masked": atom_is_masked, "bond_is_masked": bond_is_masked,
    }


def _v_has_nan(V):
    for f in ("pos", "chg", "sig", "eps"):
        if f not in V:
            continue
        v = V[f]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        if not torch.isfinite(v).all():
            return True
    return False


class RandomKPerMoleculeDataset(Dataset):
    def __init__(self, raw_dataset, k_per_molecule_per_epoch=2, restrict_k_gt=10):
        self.raw_dataset = raw_dataset
        self.k_per_molecule_per_epoch = k_per_molecule_per_epoch
        self.restrict_k_gt = restrict_k_gt
        self.index = []
        self._n_skipped_nan = 0
        self._n_dropped_mol = 0
        self.resample_index()

    def __len__(self):
        return len(self.index)

    def _valid_keys(self, sample):
        # K ranges 1..M in the raw data (K_all includes the trivial K=M case,
        # one site per atom / no compression), but training only uses
        # K=1..M-1, matching the method section and sample.py's default range.
        m_atoms = int(sample["M"])
        valid_keys = []
        for key in sample["K_keys"]:
            k_value = int(key[1:])
            if k_value <= self.restrict_k_gt:
                continue
            if k_value >= m_atoms:
                continue
            if _v_has_nan(sample["K_all"][key]):
                self._n_skipped_nan += 1
                continue
            valid_keys.append(key)
        return valid_keys

    def resample_index(self):
        self.index = []
        self._n_skipped_nan = 0
        self._n_dropped_mol = 0
        for mol_idx, sample in enumerate(self.raw_dataset):
            if "K_keys" not in sample:
                continue
            valid_keys = self._valid_keys(sample)
            if len(valid_keys) == 0:
                self._n_dropped_mol += 1
                continue
            for _ in range(self.k_per_molecule_per_epoch):
                self.index.append(mol_idx)
        random.shuffle(self.index)
        if self._n_skipped_nan > 0 or self._n_dropped_mol > 0:
            print(
                f"[Dataset] skipped {self._n_skipped_nan} NaN K representations; "
                f"dropped {self._n_dropped_mol} molecules with no valid K."
            )

    def __getitem__(self, idx):
        mol_idx = self.index[idx]
        sample = self.raw_dataset[mol_idx]
        valid_keys = self._valid_keys(sample)
        if len(valid_keys) == 0:
            m_atoms = int(sample["M"])
            valid_keys = [
                k for k in sample["K_keys"]
                if self.restrict_k_gt < int(k[1:]) < m_atoms
            ]
        key = random.choice(valid_keys)
        return {
            "AA": sample["AA"],
            "V": sample["K_all"][key],
            "M": sample["M"],
            "K_value": int(key[1:]),
            "source_file": sample.get("source_file", ""),
            "source_key": key,
        }


def make_training_batch(
    samples,
    device="cpu",
    path_noise=0.01,
    atom_mask_id=12,
    bond_mask_id=5,
):
    states = [
        make_state(
            sample, device=device,
            path_noise=path_noise, atom_mask_id=atom_mask_id, bond_mask_id=bond_mask_id,
        )
        for sample in samples
    ]
    B = len(states)
    max_M = max(s["St"].shape[0] for s in states)
    max_K = max(s["V"].shape[0] for s in states)
    batch = {}
    batch["St"] = torch.zeros(B, max_M, 4, device=device)
    batch["S1"] = torch.zeros(B, max_M, 4, device=device)
    batch["V"] = torch.zeros(B, max_K, 6, device=device)
    batch["At"] = torch.full((B, max_M), atom_mask_id, dtype=torch.long, device=device)
    batch["A"] = torch.full((B, max_M), -100, dtype=torch.long, device=device)
    batch["Bt"] = torch.full((B, max_M, max_M), bond_mask_id, dtype=torch.long, device=device)
    batch["B"] = torch.full((B, max_M, max_M), -100, dtype=torch.long, device=device)
    batch["t"] = torch.zeros(B, device=device)
    batch["atom_valid"] = torch.zeros(B, max_M, dtype=torch.bool, device=device)
    batch["site_valid"] = torch.zeros(B, max_K, dtype=torch.bool, device=device)
    batch["atom_is_masked"] = torch.zeros(B, max_M, dtype=torch.bool, device=device)
    batch["bond_is_masked"] = torch.zeros(B, max_M, max_M, dtype=torch.bool, device=device)
    for b, s in enumerate(states):
        M = s["St"].shape[0]
        K_n = s["V"].shape[0]
        batch["St"][b, :M] = s["St"]
        batch["S1"][b, :M] = s["S1"]
        batch["V"][b, :K_n] = s["V"]
        batch["At"][b, :M] = s["At"]
        batch["A"][b, :M] = s["A"]
        batch["Bt"][b, :M, :M] = s["Bt"]
        batch["B"][b, :M, :M] = s["B"]
        batch["t"][b] = s["t"].view(-1)[0]
        batch["atom_valid"][b, :M] = True
        batch["site_valid"][b, :K_n] = True
        batch["atom_is_masked"][b, :M] = s["atom_is_masked"]
        batch["bond_is_masked"][b, :M, :M] = s["bond_is_masked"]
    return batch


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def split_molecules(raw_dataset, val_frac=0.1, seed=42):
    """Molecule-level deterministic train/val split (a molecule's different K
    representations never straddle the split). Used by sampling to reproduce
    the same val set training held out."""
    n = len(raw_dataset)
    g = random.Random(seed)
    idx = list(range(n))
    g.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    train_raw = [raw_dataset[i] for i in idx[n_val:]]
    val_raw = [raw_dataset[i] for i in idx[:n_val]]
    return train_raw, val_raw
