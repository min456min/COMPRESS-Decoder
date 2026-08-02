# COMPRESS Decoder

A graph neural network decoder that reconstructs all-atom molecular structures
from COMPRESS representations (fixed-size sets of physically parameterized
spatial sites). Positions and charges are generated with flow matching;
atom types and bond orders are generated with discrete (CTMC) flow matching.

## Structure

```
configs/        hyperparameter configs (YAML)
data/           input data (data.pt, val.pt, ...) -- not tracked, place your files here
checkpoints/    saved model checkpoints -- not tracked, created during training
outputs/        sample.py / example.py results -- not tracked, created at runtime
decoder/        library code
  model.py        the COMPRESSDecoder network and its building blocks
  data.py         data prep, batching, and the training Dataset
  losses.py       loss functions
  utils.py        checkpointing and logging
  compress_io.py  reads raw output from the COMPRESS site-optimization tool
train.py        training entry point
sample.py       sampling / inference entry point
example.py      minimal end-to-end example
```

## Setup

```bash
pip install -r requirements.txt
```

Place your training data at `data/data.pt` (see `decoder/data.py` for the
expected format: a list of molecule entries, each with `AA`, `K_all`, `M`,
and `K_keys`).

## Data

`data.pt`: **https://doi.org/10.5281/zenodo.21764574**. Download it and
place it at `data/data.pt` before training.

`data/val.pt` (the held-out validation split) is included directly in this
repository.

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
# Generate from the held-out validation molecules included in this repo
python sample.py --mode val --ckpt checkpoints/latest.pt

# Generate straight from raw output of the COMPRESS site-optimization tool
# (https://github.com/ADicksonLab/COMPRESS): point at the folder containing
# that tool's "{name}_s{K}_COMPRESS.pt" files for one molecule.
python sample.py --mode notarget --ckpt checkpoints/latest.pt --compress_dir path/to/compress_output --name test
```

Also supports `--mode notarget --k_file ...` (a pre-bundled COMPRESS dict);
see `python sample.py --help`.

Each generated molecule is saved under `--out` (default `outputs/`) with a
`{tag}_K{n}_try{t}_...` filename prefix (`tag` e.g. `val_mol0`, `target`,
`notarget_M30`). Per (molecule, K, try), you get:

- `..._sites.mol2` - the COMPRESS representation V (its K sites), as a point cloud
- `..._init.mol2` - the initial atom cloud S0, before refinement
- `..._result.mol2` - the generated molecule
- `..._result_attr.csv` - per-atom element, position, and charge of the result
- `{tag}_target.mol2` - the ground-truth molecule (val modes only)


### Quick example

```bash
python example.py
```

Loads the first molecule in `data/val.pt`, reconstructs it from its K=25 COMPRESS representation, and writes the result to `outputs/example/`.

## Config
All model, training, and vocabulary hyperparameters live in `configs/decoder.yaml`. Edit this file (or pass `--config` with a different path) rather than changing values in `train.py`.
