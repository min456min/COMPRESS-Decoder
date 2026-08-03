# COMPRESS Decoder

A flow matching model that reconstructs all-atom molecular structures from COMPRESS representations (fixed-size sets of physically parameterized spatial sites).

## Structure

```
configs/        hyperparameter configs
data/           input data - val.pt and data.pt
checkpoints/    saved model checkpoints 
outputs/        sample.py / example.py results
decoder/        library code
  model.py        the COMPRESSDecoder network and its building blocks
  data.py         data prep, batching, and the training Dataset
  losses.py       loss functions
  utils.py        checkpointing and logging
  compress_io.py  reads raw output from the COMPRESS site-optimization tool
train.py        main training script
sample.py       sampling / inference entry point
example.py      minimal example
check_data.py   check a data.pt / val.pt file's schema
```

## Setup

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate compress-decoder
```

## Data

`data.pt`: **https://doi.org/10.5281/zenodo.21764574**. 
Download it and place it at `data/data.pt` before training.

It's a list of Python dicts, one per molecule/conformer, each with:

- `AA`: all-atom reference - `pos` (M,3), `chg` (M,), `atom_type` (M,) int (0=H, 1=C, 2=N, 3=O, 4=S, 5=P, 6=F, 7=Cl, 8=Br, 9=I, 10=other), `bond_type` (M,M) int (0=none, 1-4=bond order incl. aromatic)
- `K_all`: dict `{"K1": {...}, ..., "K{M}": {...}}`, each holding `pos` (K,3), `chg` (K,), `sig` (K,), `eps` (K,) -- COMPRESS site position, charge,Lennard-Jones radius, and well depth
- `K_keys`: list of valid `"K{n}"` keys for this molecule
- `M`: atom count

```bash
python check_data.py data/data.pt
```
Validates the file's schema and prints atom-count stats.

`data/val.pt` (the held-out validation split) is included directly in this repository.

## Training

```bash
python train.py --config configs/decoder.yaml
```

For multi-GPU (or single-GPU) distributed training:

```bash
torchrun --nproc_per_node=<N> train.py --config configs/decoder.yaml
```


## Sampling

```bash
# Generate from the held-out validation molecules included in this repo python sample.py --mode val --ckpt checkpoints/latest.pt

# Generate straight from raw output of the COMPRESS site-optimization tool
# (https://github.com/ADicksonLab/COMPRESS): point at the folder containing
# that tool's "{name}_s{K}_COMPRESS.pt" files for one molecule.

python sample.py --mode notarget --ckpt checkpoints/latest.pt --compress_dir path/to/compress_output --name test
```

Also supports `--mode notarget --k_file ...` (a pre-bundled COMPRESS dict);
see `python sample.py --help`.

Each generated molecule is saved under `--out` (default `outputs/`) with a
`{tag}_K{n}_try{t}_...` filename prefix (`tag` e.g. `val_mol0`, `notarget_M30`).
Per (molecule, K, try), you get:

- `..._sites.mol2` - the COMPRESS representation V (its K sites), as a point cloud
- `..._sites_attr.csv` -- per-site position, charge, sigma, and epsilon of V  
- `..._init.mol2` - the initial atom cloud S0, before refinement
- `..._result.mol2` - the generated molecule
- `{tag}_target.mol2` - the ground-truth molecule (val/target modes only)


### Quick example

```bash
python example.py
```

Loads the first molecule in `data/val.pt`, reconstructs it from its K=25 COMPRESS representation, and writes the result to `outputs/example/`.

## Config
All model, training, and vocabulary hyperparameters live in `configs/decoder.yaml`. Edit this file rather than changing values in `train.py`.
