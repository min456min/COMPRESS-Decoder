"""Generate the first molecule in data/val.pt from its K=25 representation.

Run: python example.py
"""
import torch

from sample import load_decoder, generate_for_entry

CKPT = "checkpoints/latest.pt"
VAL_FILE = "data/val.pt"
OUT_DIR = "outputs/example"
K_TARGET = 25


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_raw = torch.load(VAL_FILE, map_location="cpu")
    entry = val_raw[0]
    M = int(entry["M"])

    available_ks = sorted(int(k[1:]) for k in entry["K_all"])
    k_value = max([k for k in available_ks if k <= K_TARGET], default=available_ks[0])
    print(f"Molecule 0: M={M} atoms, {len(available_ks)} K options available; using K={k_value}")

    decoder = load_decoder(CKPT, device)
    print(f"Loaded decoder: {CKPT}")

    generate_for_entry(
        decoder, entry["K_all"], M, device,
        out_root=OUT_DIR, tag="example_mol0",
        k_min=k_value, k_max=k_value, n_try=1, n_step=300,
        target_aa=entry["AA"],
    )
    print(f"Done. See {OUT_DIR}/example_mol0_K{k_value}_try1_result.mol2 "
          f"(plus _sites.mol2, _init.mol2, _result_attr.csv, and _target.mol2 "
          f"for the ground truth).")


if __name__ == "__main__":
    main()
