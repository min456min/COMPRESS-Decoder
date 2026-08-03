"""
Sampling / inference for the COMPRESS decoder.

V = COMPRESS representation (K sites), S = all-atom structure (M atoms),
per the method section. K/M are still used for the scalar site/atom counts.

Modes:
  val      : validation molecules from data.pt, every K in K1..K(M-1).
  notarget : only V is known; M is supplied by the user.

Outputs under --out, per (molecule, K, try):
  {tag}_target.mol2                  ground truth (val mode only)
  {tag}_K{n}_sites.mol2               COMPRESS sites V
  {tag}_K{n}_try{t}_init.mol2         initial atom cloud S0
  {tag}_K{n}_try{t}_result.mol2       generated molecule
  {tag}_K{n}_try{t}_result_attr.csv   per-atom element/position/charge
"""
import os
import re
import argparse
import numpy as np
import torch

from decoder.model import COMPRESSDecoder
from decoder.data import make_V, make_S0, split_molecules
from decoder.compress_io import load_k_all_from_compress_dir

# ============================================================
# Vocab (matches decoder/data.py)
# ============================================================
# 11 = unused slot; 12 = MASK.
ATOM_ID_TO_ELEMENT = {
    0: "H", 1: "C", 2: "N", 3: "O", 4: "S", 5: "P",
    6: "F", 7: "Cl", 8: "Br", 9: "I",
    10: "C",  # other
    11: "C",  # unused slot
    12: "C",  # MASK
}
ATOM_MASK_ID = 12
BOND_MASK_ID = 5
BOND_ID_TO_MOL2 = {1: "1", 2: "2", 3: "3", 4: "ar"}


# ============================================================
# Prediction -> discrete conversion
# ============================================================
def atom_logits_to_ids(atom_logits):
    return atom_logits.argmax(dim=-1)[0].detach().cpu().numpy()


# ============================================================
# CTMC discrete sampling
# ============================================================
def _purity_unmask(xt, x1_probs, unmask_prob, mask_index, hc_thresh, device):
    masked = (xt == mask_index)
    n_masked = int(masked.sum())
    if n_masked == 0:
        return torch.zeros_like(xt, dtype=torch.bool)
    purities = x1_probs.max(dim=-1).values          # top prob per token
    hc = (purities >= hc_thresh) & masked
    n_hc = int(hc.sum())
    node_unmask_prob = torch.zeros(xt.shape[0], device=device)
    if n_hc == 0:
        node_unmask_prob[masked] = float(unmask_prob)
    else:
        ph_max = unmask_prob * n_masked / n_hc
        ph = min(ph_max, 1.0)
        denom = (n_masked - n_hc)
        pl = (unmask_prob * n_masked - ph * n_hc) / denom if denom > 0 else 0.0
        pl = max(0.0, pl)
        node_unmask_prob[hc] = ph
        lc = masked & (~hc)
        node_unmask_prob[lc] = pl
    return torch.rand(xt.shape[0], device=device) < node_unmask_prob


def campbell_step_tokens(xt, p1, t_i, dt, mask_index,
                         eta=40.0, hc_thresh=0.9, last_step=False, device="cpu"):
    xt = xt.clone()
    x1 = torch.distributions.Categorical(probs=p1).sample()   # (n,) in [0, n_real)
    alpha_t = max(float(t_i), 0.0)
    denom = max(1.0 - alpha_t, 1e-3)
    unmask_prob = dt * (1.0 + eta * alpha_t) / denom      # alpha_t'=1 (linear)
    mask_prob = dt * eta
    unmask_prob = float(min(max(unmask_prob, 0.0), 1.0))
    mask_prob = float(min(max(mask_prob, 0.0), 1.0))
    if hc_thresh > 0:
        will_unmask = _purity_unmask(xt, p1, unmask_prob, mask_index, hc_thresh, device)
    else:
        will_unmask = (torch.rand(xt.shape[0], device=device) < unmask_prob) & (xt == mask_index)
    if not last_step:
        will_mask = (torch.rand(xt.shape[0], device=device) < mask_prob) & (xt != mask_index)
        xt[will_mask] = mask_index
    xt[will_unmask] = x1[will_unmask]
    return xt


# ============================================================
# Writers
# ============================================================
def write_mol2(S_pred, atom_ids, bond_type, path, mol_name="CG"):
    """Write a molecule as MOL2."""
    S_pred = np.asarray(S_pred)
    coords = S_pred[:, :3]
    charges = S_pred[:, 3]
    N = coords.shape[0]
    elems = [ATOM_ID_TO_ELEMENT.get(int(a), "C") for a in atom_ids]
    bonds = []
    bid = 1
    for i in range(N):
        for j in range(i + 1, N):
            bt = int(bond_type[i, j])
            if bt == 0:
                continue
            bonds.append((bid, i + 1, j + 1, BOND_ID_TO_MOL2.get(bt, "1")))
            bid += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{mol_name}\n")
        f.write(f"{N:5d} {len(bonds):5d}     1     0     0\n")
        f.write("SMALL\nUSER_CHARGES\n\n")
        f.write("@<TRIPOS>ATOM\n")
        for i in range(N):
            x, y, z = coords[i]
            elem = elems[i]
            f.write(
                f"{i + 1:7d} {elem}{i + 1:<4d} "
                f"{x:10.4f} {y:10.4f} {z:10.4f} "
                f"{elem:<6s} {1:4d} MOL {float(charges[i]):10.6f}\n"
            )
        f.write("@<TRIPOS>BOND\n")
        for bid, i, j, bt in bonds:
            f.write(f"{bid:6d} {i:5d} {j:5d} {bt:>2s}\n")
        f.write("@<TRIPOS>SUBSTRUCTURE\n")
        f.write("     1 MOL         1 TEMP              0 ****  ****    0 ROOT\n")
    return path


def write_target_mol2(aa, path, mol_name="target"):
    """Write the target molecule from an AA dict """
    pos = np.asarray(aa["pos"])
    chg = np.asarray(aa["chg"]).reshape(-1)
    atom_type = np.asarray(aa["atom_type"]).reshape(-1).astype(int)
    bond_type = np.asarray(aa["bond_type"]).astype(int)
    N = pos.shape[0]
    elems = [ATOM_ID_TO_ELEMENT.get(int(a), "C") for a in atom_type]
    bonds = []
    bid = 1
    for i in range(N):
        for j in range(i + 1, N):
            bt = int(bond_type[i, j])
            if bt == 0:
                continue
            bonds.append((bid, i + 1, j + 1, BOND_ID_TO_MOL2.get(bt, "1")))
            bid += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{mol_name}\n")
        f.write(f"{N:5d} {len(bonds):5d}     1     0     0\n")
        f.write("SMALL\nUSER_CHARGES\n\n")
        f.write("@<TRIPOS>ATOM\n")
        for i in range(N):
            x, y, z = pos[i]
            elem = elems[i]
            f.write(
                f"{i + 1:7d} {elem}{i + 1:<4d} "
                f"{x:10.4f} {y:10.4f} {z:10.4f} "
                f"{elem:<6s} {1:4d} MOL {float(chg[i]):10.6f}\n"
            )
        f.write("@<TRIPOS>BOND\n")
        for bid, i, j, bt in bonds:
            f.write(f"{bid:6d} {i:5d} {j:5d} {bt:>2s}\n")
        f.write("@<TRIPOS>SUBSTRUCTURE\n")
        f.write("     1 MOL         1 TEMP              0 ****  ****    0 ROOT\n")
    return path


def write_init_mol2(S0, path, mol_name="init"):
    """Write the initial S0 cloud"""
    S0 = np.asarray(S0)
    coords = S0[:, :3]
    charges = S0[:, 3]
    N = coords.shape[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{mol_name}\n")
        f.write(f"{N:5d} {0:5d}     1     0     0\n")
        f.write("SMALL\nUSER_CHARGES\n\n")
        f.write("@<TRIPOS>ATOM\n")
        for i in range(N):
            x, y, z = coords[i]
            f.write(
                f"{i + 1:7d} C{i + 1:<4d} "
                f"{x:10.4f} {y:10.4f} {z:10.4f} "
                f"{'C':<6s} {1:4d} MOL {float(charges[i]):10.6f}\n"
            )
        f.write("@<TRIPOS>BOND\n")
        f.write("@<TRIPOS>SUBSTRUCTURE\n")
        f.write("     1 MOL         1 TEMP              0 ****  ****    0 ROOT\n")
    return path


 
def write_site_attr_csv(V_data, path):
    """Write per-site COMPRESS representation attributes (position, charge, sigma, epsilon) as CSV."""
    pos = np.asarray(V_data["pos"])
    chg = np.asarray(V_data["chg"]).reshape(-1)
    sig = np.asarray(V_data["sig"]).reshape(-1)
    eps = np.asarray(V_data["eps"]).reshape(-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("site_index,x,y,z,charge,sigma,epsilon\n")
        for i in range(pos.shape[0]):
            x, y, z = pos[i]
            f.write(f"{i},{x:.4f},{y:.4f},{z:.4f},"
                    f"{float(chg[i]):.6f},{float(sig[i]):.6f},{float(eps[i]):.6f}\n")
    return path
 
 
# ============================================================
# Model loading
# ============================================================
def load_decoder(checkpoint_path, device):
    """Build a COMPRESSDecoder from the checkpoint's saved config"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    vocab_cfg = cfg.get("vocab", {})
 
    def g(d, k, default):
        return d.get(k, default)
 
    decoder = COMPRESSDecoder(
        hidden_dim=g(model_cfg, "hidden_dim", 256), k_hidden_dim=g(model_cfg, "k_hidden_dim", 256),
        edge_dim=g(model_cfg, "edge_dim", 128), time_dim=g(model_cfg, "time_dim", 64),
        atom_emb_dim=g(model_cfg, "atom_emb_dim", 64), bond_emb_dim=g(model_cfg, "bond_emb_dim", 64),
        n_rbf=g(model_cfg, "n_rbf", 48), rbf_dmax=g(model_cfg, "rbf_dmax", 12.0),
        n_blocks=g(model_cfg, "n_blocks", 6), k_blocks=g(model_cfg, "k_blocks", 3),
        num_atom_types=g(vocab_cfg, "num_atom_types", 13), num_bond_types=g(vocab_cfg, "num_bond_types", 6),
        num_atom_classes=g(vocab_cfg, "num_atom_classes", 12), num_bond_classes=g(vocab_cfg, "num_bond_classes", 5),
    ).to(device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    decoder.load_state_dict(state)
    decoder.eval()
    return decoder
 
 
# ============================================================
# Iterative sampling
# ============================================================
@torch.no_grad()
def iterative_sample(decoder, V, device, M_atoms,
                     n_step=300, t_start=0.0, t_end=1.0,
                     remove_com=True, eta=40.0, hc_thresh=0.9,
                     make_s0_kwargs=None):
    """
    Generate one molecule (S) from a COMPRESS representation V.
    Continuous features (position, charge): flow-matching Euler integration, vf = (x_1_pred - x_t)/(1-t). 
    Discrete features (atom type, bond type): CTMC sampling (see campbell_step_tokens).
    Args:
        V        : (K_n, 6) COMPRESS tensor from make_V.
        M_atoms  : number of atoms to generate (int).
        Returns  : (S0_np, S_final_np, atom_ids, bond_type)
    """
    make_s0_kwargs = make_s0_kwargs or dict(
        min_dist=0.3, max_tries=1000, chg_noise_scale=0.3, pos_noise_scale=0.5,
    )
    # Center the V sites at the origin before building S0 (the model was
    # trained on centered coordinates); add this COM back at the end.
    v_com = V[:, :3].mean(dim=0, keepdim=True)        # (1, 3)
    V = V.clone()
    V[:, :3] = V[:, :3] - v_com
    v_com_np = v_com.squeeze(0).detach().cpu().numpy()   # (3,)
 
    dummy_S = torch.zeros(M_atoms, 4, device=device)
    S0 = make_S0(dummy_S, V, device=device, **make_s0_kwargs)   # (M_atoms, 4)
    M_n = S0.shape[0]
    K_n = V.shape[0]
    St = S0.unsqueeze(0)          # (1, M, 4)
    V_b = V.unsqueeze(0)          # (1, K_n, 6)
 
    At_tok = torch.full((M_n,), ATOM_MASK_ID, dtype=torch.long, device=device)
    tri_i, tri_j = torch.triu_indices(M_n, M_n, offset=1, device=device)
    Bt_tok = torch.full((tri_i.shape[0],), BOND_MASK_ID, dtype=torch.long, device=device)
    atom_valid = torch.ones(1, M_n, dtype=torch.bool, device=device)
    site_valid = torch.ones(1, K_n, dtype=torch.bool, device=device)
 
    S0_np = S0.detach().cpu().numpy()
    final_atom_logits = None
    final_bond_logits = None
 
    t_grid = torch.linspace(t_start, t_end, n_step + 1, device=device)
 
    def _atom_tokens_to_batch(tok):
        return tok.unsqueeze(0)                      # (1, M)
 
    def _bond_tokens_to_batch(tok):
        Bt = torch.full((M_n, M_n), BOND_MASK_ID, dtype=torch.long, device=device)
        d = torch.arange(M_n, device=device)
        Bt[d, d] = 0
        Bt[tri_i, tri_j] = tok
        Bt[tri_j, tri_i] = tok
        return Bt.unsqueeze(0)                       # (1, M, M)
 
    for step in range(n_step):
        t_i = float(t_grid[step])
        t_next = float(t_grid[step + 1])
        dt = t_next - t_i
        last_step = (step == n_step - 1)
 
        t = torch.full((1,), t_i, dtype=St.dtype, device=device)
        At = _atom_tokens_to_batch(At_tok)
        Bt = _bond_tokens_to_batch(Bt_tok)
 
        S1_pred, atom_logits, bond_logits = decoder(
            St=St, At=At, Bt=Bt, t=t, V=V_b,
            atom_valid=atom_valid, site_valid=site_valid,
        )
        atom_p1 = torch.softmax(atom_logits[0], dim=-1)
        bond_p1_full = torch.softmax(bond_logits[0], dim=-1)
 
        # ---- continuous: flow-matching Euler step ----
        denom = max(1.0 - t_i, 1e-3)
        vf = (S1_pred - St) / denom
        St = St + vf * dt
        if remove_com:
            com = St[:, :, :3].mean(dim=1, keepdim=True)
            St[:, :, :3] = St[:, :, :3] - com
 
        # ---- discrete atoms: CTMC campbell step ----
        At_tok = campbell_step_tokens(
            At_tok, atom_p1, t_i, dt, mask_index=ATOM_MASK_ID,
            eta=eta, hc_thresh=hc_thresh,
            last_step=last_step, device=device,
        )
        # ---- discrete bonds: CTMC on the upper triangle ----
        bond_p1_upper = bond_p1_full[tri_i, tri_j]               # (n_pairs, num_bond_classes)
        Bt_tok = campbell_step_tokens(
            Bt_tok, bond_p1_upper, t_i, dt, mask_index=BOND_MASK_ID,
            eta=eta, hc_thresh=hc_thresh,
            last_step=last_step, device=device,
        )
        final_atom_logits = atom_logits
        final_bond_logits = bond_logits
 
    S_final = St[0].detach().cpu().numpy()
    S_final[:, :3] = S_final[:, :3] + v_com_np[None, :]
 
    atom_ids = At_tok.detach().cpu().numpy()
    still = atom_ids == ATOM_MASK_ID
    if still.any():
        fb = atom_logits_to_ids(final_atom_logits)
        atom_ids[still] = fb[still]
 
    bond_type = np.zeros((M_n, M_n), dtype=np.int64)
    bt_up = Bt_tok.detach().cpu().numpy()
    ti = tri_i.detach().cpu().numpy()
    tj = tri_j.detach().cpu().numpy()
    if (bt_up == BOND_MASK_ID).any():
        bl_pred = final_bond_logits[0].argmax(dim=-1).detach().cpu().numpy()
        for p in np.nonzero(bt_up == BOND_MASK_ID)[0]:
            bt_up[p] = bl_pred[ti[p], tj[p]]
    bond_type[ti, tj] = bt_up
    bond_type[tj, ti] = bt_up
 
    return S0_np, S_final, atom_ids, bond_type
 
 
# ============================================================
# Generation over K1..K(M-1) for one molecule entry
# ============================================================
def generate_for_entry(decoder, K_all, M_atoms, device, out_root, tag,
                       k_min=1, k_max=None, n_try=1,
                       n_step=300, target_aa=None, eta=40.0, hc_thresh=0.9):
    if k_max is None:
        k_max = M_atoms - 1
    outdir = out_root
    os.makedirs(outdir, exist_ok=True)
 
    if target_aa is not None:
        write_target_mol2(target_aa, os.path.join(outdir, f"{tag}_target.mol2"),
                          mol_name=f"{tag}_target")
 
    for k_value in range(k_min, k_max + 1):
        key = f"K{k_value}"
        if key not in K_all:
            continue
        V_data = {kk: K_all[key][kk] for kk in ("pos", "chg", "sig", "eps")}
        V = make_V(V_data, device=device)
        write_init_mol2(V[:, :4].cpu().numpy(),
                         os.path.join(outdir, f"{tag}_K{k_value}_sites.mol2"),
                         mol_name=f"{tag}_K{k_value}_sites")
        write_site_attr_csv(V_data, os.path.join(outdir, f"{tag}_K{k_value}_sites_attr.csv"))
        for try_idx in range(1, n_try + 1):
            S0_np, S_final, atom_ids, bond_type = iterative_sample(
                decoder, V, device, M_atoms=M_atoms,
                n_step=n_step, eta=eta, hc_thresh=hc_thresh,
            )
            if target_aa is not None:
                v_com_np = V[:, :3].mean(dim=0).detach().cpu().numpy()
                target_com_np = np.asarray(target_aa["pos"]).mean(axis=0)
                offset = target_com_np - v_com_np
                S0_np = S0_np.copy()
                S0_np[:, :3] += offset
                S_final = S_final.copy()
                S_final[:, :3] += offset
            stem = f"{tag}_K{k_value}_try{try_idx}"
            write_init_mol2(S0_np, os.path.join(outdir, f"{stem}_init.mol2"),
                             mol_name=f"{stem}_init")
            write_mol2(S_final, atom_ids, bond_type,
                       os.path.join(outdir, f"{stem}_result.mol2"),
                       mol_name=f"{stem}_result")
            print(f"  [{tag}] K{k_value} try{try_idx}: {M_atoms} atoms")
 
 
# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="COMPRESS decoder sampling")
    p.add_argument("--mode", required=True, choices=["val", "notarget"])
    p.add_argument("--ckpt", default="checkpoints/latest.pt",
                   help="model checkpoint (saved by train.py)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--n_step", type=int, default=300)
    p.add_argument("--eta", type=float, default=40.0,
                   help="CTMC stochasticity (higher = more re-masking/exploration)")
    p.add_argument("--hc_thresh", type=float, default=0.9,
                   help="CTMC high-confidence threshold for purity unmasking")
    p.add_argument("--n_try", type=int, default=1)
    p.add_argument("--m_offset_min", type=int, default=0,
                   help="lowest atom-count offset (val mode). One run generates "
                        "every offset from m_offset_min..m_offset_max.")
    p.add_argument("--m_offset_max", type=int, default=0,
                   help="highest atom-count offset (inclusive). Must be >= m_offset_min.")
    # mode=val
    p.add_argument("--data", default="data/data.pt", help="data.pt (for mode=val)")
    p.add_argument("--val_cache", default="data/val.pt",
                   help="cached val split; loaded if present, else built from --data and saved here")
    p.add_argument("--n_mol", type=int, default=3, help="how many val molecules (mode=val)")
    p.add_argument("--shuffle", action="store_true",
                   help="pick n_mol RANDOM val molecules instead of the first n "
                        "(so each run generates different molecules).")
    p.add_argument("--sample_seed", type=int, default=None,
                   help="seed for --shuffle selection. Omit for a different draw each run; "
                        "set an int to reproduce the same random selection.")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # mode=notarget
    p.add_argument("--k_file", default=None,
                   help="pre-bundled dict with K_all (mode=notarget). Alternative to "
                        "--compress_dir; give one or the other.")
    p.add_argument("--compress_dir", default=None,
                   help="folder of raw COMPRESS.py output ('{name}_s{K}_COMPRESS.pt' files) "
                        "for one molecule (mode=notarget). Lets you go straight from "
                        "'compress ...' output to sampling with no manual conversion step.")
    p.add_argument("--name", default=None,
                   help="molecule name to filter on within --compress_dir, if the folder "
                        "holds output for more than one molecule.")
    p.add_argument("--n_atoms", type=int, default=None,
                   help="number of atoms M to generate (mode=notarget). Use this for a "
                        "single M, OR use --n_atoms_min/--n_atoms_max for a range. If "
                        "omitted with --compress_dir, M is auto-detected from AA_pos.")
    p.add_argument("--n_atoms_min", type=int, default=None,
                   help="lowest M (mode=notarget). With --n_atoms_max, generates every "
                        "M in [min, max] in one run, e.g. --n_atoms_min 21 --n_atoms_max 50.")
    p.add_argument("--n_atoms_max", type=int, default=None,
                   help="highest M (inclusive). Must be >= n_atoms_min.")
    args = p.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | mode: {args.mode}")
 
    decoder = load_decoder(args.ckpt, device)
    print(f"Loaded decoder: {args.ckpt}")
 
    os.makedirs(args.out, exist_ok=True)
 
    if args.mode == "val":
        if os.path.exists(args.val_cache):
            val_raw = torch.load(args.val_cache, map_location="cpu")
            print(f"Loaded cached val split: {args.val_cache} ({len(val_raw)} molecules)")
        else:
            print(f"No cache at {args.val_cache}; building val split from {args.data} ...")
            raw = torch.load(args.data, map_location="cpu")
            _, val_raw = split_molecules(raw, val_frac=args.val_frac, seed=args.seed)
            torch.save(val_raw, args.val_cache)
            print(f"Saved val split -> {args.val_cache} ({len(val_raw)} molecules)")
        n = min(args.n_mol, len(val_raw))
        offsets = list(range(args.m_offset_min, args.m_offset_max + 1))
        if args.shuffle:
            import random as _r
            rng = _r.Random(args.sample_seed)  # sample_seed=None -> different every run
            indices = rng.sample(range(len(val_raw)), n)
        else:
            indices = list(range(n))
        print(f"Generating for {n} val molecules (indices {indices}); offsets {offsets}.")
        for mi in indices:
            entry = val_raw[mi]
            M_target = int(entry["M"])
            for off in offsets:
                M_atoms = max(1, M_target + off)
                off_tag = "" if off == 0 else f"_off{off:+d}"
                generate_for_entry(
                    decoder, entry["K_all"], M_atoms, device,
                    out_root=args.out, tag=f"val_mol{mi}{off_tag}",
                    n_try=args.n_try, n_step=args.n_step, eta=args.eta, hc_thresh=args.hc_thresh,
                    target_aa=entry["AA"],
                )
    elif args.mode == "notarget":
        M_detected = None
        if args.compress_dir:
            K_all, M_detected = load_k_all_from_compress_dir(args.compress_dir, name=args.name)
            print(f"Loaded {len(K_all)} K representations from {args.compress_dir}"
                  + (f" (name={args.name})" if args.name else ""))
        elif args.k_file:
            data = torch.load(args.k_file, map_location="cpu")
            if isinstance(data, list):
                data = data[0]
            K_all = data["K_all"] if "K_all" in data else {
                k: data[k] for k in data if re.fullmatch(r"K\d+", str(k))
            }
        else:
            raise ValueError("mode=notarget requires --compress_dir or --k_file.")
 
        if args.n_atoms_min is not None and args.n_atoms_max is not None:
            m_list = list(range(args.n_atoms_min, args.n_atoms_max + 1))
        elif args.n_atoms is not None:
            m_list = [args.n_atoms]
        elif M_detected is not None:
            m_list = [M_detected]
            print(f"No --n_atoms given; using M={M_detected} auto-detected from COMPRESS output.")
        else:
            raise ValueError("mode=notarget requires --n_atoms OR (--n_atoms_min and "
                             "--n_atoms_max), or --compress_dir output with AA_pos to "
                             "auto-detect M.")
        print(f"No-target mode: M values {m_list}, K sizes available: {len(K_all)}.")
        for M_atoms in m_list:
            generate_for_entry(
                decoder, K_all, M_atoms, device,
                out_root=args.out, tag=f"notarget_M{M_atoms}",
                n_try=args.n_try, n_step=args.n_step, eta=args.eta, hc_thresh=args.hc_thresh,
            )
    print("Done.")
 
 
if __name__ == "__main__":
    main()
