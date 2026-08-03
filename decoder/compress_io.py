"""
Read raw output from the COMPRESS site-optimization tool (https://github.com/ADicksonLab/COMPRESS), 
so its .pt files can be fed directly into sample.py without a separate manual conversion step.

COMPRESS.py writes one file per (molecule, K): "{name}_s{K}_COMPRESS.pt",
each holding that K's site parameters (pos, chg, sig, eps) plus the all-atom reference (AA_pos, AA_chg, AA_sig, AA_eps). 
This module bundles however many K files exist for one molecule into the K_all dict our decoder expects.
"""
import os
import re
import glob
import torch

_FILE_RE = re.compile(r"^(?P<name>.+)_s(?P<k>\d+)_COMPRESS\.pt$")


def load_k_all_from_compress_dir(compress_dir, name=None):
    """
    Scan `compress_dir` for COMPRESS.py output files and bundle them into 
    K_all = {"K1": {"pos","chg","sig","eps"}, "K2": {...}, ...}.
    """
    K_all = {}
    M = None
    for path in sorted(glob.glob(os.path.join(compress_dir, "*_s*_COMPRESS.pt"))):
        m = _FILE_RE.match(os.path.basename(path))
        if m is None:
            continue
        if name is not None and m.group("name") != name:
            continue
        k = int(m.group("k"))
        data = torch.load(path, map_location="cpu")
        K_all[f"K{k}"] = {
            "pos": data["pos"],
            "chg": data["chg"],
            "sig": data["sig"],
            "eps": data["eps"],
        }
        if M is None and "AA_pos" in data:
            M = int(data["AA_pos"].shape[0])
    if not K_all:
        raise FileNotFoundError(
            f"No COMPRESS output files (*_s*_COMPRESS.pt) found in {compress_dir}"
            + (f" for molecule '{name}'" if name else "")
        )
    return K_all, M
