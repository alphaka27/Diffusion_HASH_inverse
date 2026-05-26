# Diffusion HASH inverse
Finding the hash inverse using a diffusion model

# Environment Setting
``` bash
pip install -r requirements.txt
pip install -e .
```

# How to Run
``` bash
cd src/diffusion_hash_inv
python hash_main.py
```

# Full Workflow
See `Docs/Workflow.md` for the end-to-end workflow from hash trace generation
to RGB image dataset creation, analysis, and DDPM training.
For HDF5 tensor shard generation and PyTorch loading, see
`Docs/HDF5TensorDataset.md`.

# Log to Image and HDF5 Output
When `--make-image` is enabled, the hash run writes PNG images under
`data/images/<log-stem>/` and HDF5 tensor shards under
`data/tensor_datasets/hash_tensors/`.
Each PNG is reopened immediately after it is written and decoded in the same
writer worker to verify that it still maps back to the original log value.

In the Python API, `MainEP.run()` can split the pipeline into separate stages:
`run_hash_json=True` generates binary inputs and JSON traces, `run_png=True`
generates PNG images from existing JSON traces, and `run_hdf5=True` generates
HDF5 shards. To regenerate only PNG/HDF5 outputs from existing JSON, keep
`clean_flag=False` and call:

``` python
entrypoint.run(run_hash_json=False, run_png=True, run_hdf5=True)
```

For one artifact type only, set the other artifact flag to `False`.

`HDF5Maker` writes process-parallel tensor shards under
`data/tensor_datasets/<name>/`. Each shard stores tensors as `C,H,W` arrays and
`manifest.json` records shard counts. By default it writes every generated image
tensor from each log as `uint8` values in the `[0, 255]` range; pass
`include_paths=("message.png",)` to keep only message images. Shards also
preserve the encoded log hierarchy under
`logs/<source-log>/Message/Hex` and `logs/<source-log>/Logs/...`.

Use `HDF5TensorDataset` or `create_hdf5_tensor_dataloader` to read those shards
directly in PyTorch:

``` python
from diffusion_hash_inv.utils import create_hdf5_tensor_dataloader

loader = create_hdf5_tensor_dataloader(
    "data/tensor_datasets/hash_tensors",
    batch_size=64,
    num_workers=4,
    include_paths=("message.png",),
)
```

The HDF5 loader expects tensors in a batch to have the same shape. For training,
prefer `include_paths=("message.png",)` or another single image type.

CLI example:

``` bash
python -m diffusion_hash_inv.hash_main \
  --iteration 1000 \
  --make-image \
  --image-workers 4
```

To generate PNG images with the cube-id RGB encoding:

``` bash
python -m diffusion_hash_inv.hash_main \
  --iteration 1000 \
  --length 128 \
  --hash-alg md5 \
  --rgb-encoding cube-id \
  --make-png
```

To generate artifacts from existing JSON logs in CLI mode, skip the hash/json
stage and choose the artifact type:

``` bash
python -m diffusion_hash_inv.hash_main --skip-hash-json -l 128 --make-png
python -m diffusion_hash_inv.hash_main --skip-hash-json -l 128 --make-hdf5
```

# MLX Conditional Diffusion Example
``` bash
pip install -e ".[mlx]"
python -m diffusion_hash_inv.models.diffusion_with_mlx \
  --device cpu \
  --train-steps 200 \
  --timesteps 50 \
  --output output/conditional_diffusion_mlx_samples.png
```
`--device gpu` can be used on an Apple Silicon machine with Metal available.

# Generated Image Conditional DDPM Training
Train a conditional DDPM on PNG files generated under `data/images`.
The trainer uses only `data/images/<run-id>/message.png` files as input images.
Condition labels are read from the matching JSON file under `output/json` and
use the final hash value for each `<run-id>`.
Default `fit_mode` is `reshape`: each `message.png` is flattened and reshaped
to an equal-area square (e.g. `7168x28 -> 448x448`). `height-flatten` uses
`ImgConfig.img_size` as the block unit, represents each block as one `1x1`
pixel, and then reshapes those pixels to a square RGB image. The source
dimensions must be multiples of
`ImgConfig.img_size` (`28x28` by default). For images generated with
`--rgb-encoding cube-id`, use `--fit-mode cube-id-grid --channels 3`; this keeps
one center RGB pixel per cube-id block and makes sample decode comparisons use
the cube-id decoder. `pad` and `resize` are also available.

``` bash
pip install -e ".[train]"
python -m diffusion_hash_inv.models.conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion \
  --image-size 64 \
  --label-source final-hash \
  --batch-size 32 \
  --epochs 1 \
  --timesteps 200 \
  --beta-schedule linear \
  --save-train-batches-every 5 \
  --save-process-traces \
  --trace-sample-count 4 \
  --trace-steps 8 \
  --device auto
```

For a quick smoke run:
``` bash
python -m diffusion_hash_inv.models.conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --max-images 256 \
  --image-size 32 \
  --batch-size 8 \
  --train-steps 2 \
  --timesteps 4 \
  --base-channels 8 \
  --time-dim 16 \
  --beta-schedule linear \
  --save-train-batches-every 5 \
  --device cpu
```

Cube-id encoded PNGs can be generated and trained with:

``` bash
python -m diffusion_hash_inv.hash_main -i 10000 -l 128 --hash-alg md5 --rgb-encoding cube-id --make-png
python -m diffusion_hash_inv.models.conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_cube_id \
  --fit-mode cube-id-grid \
  --channels 3 \
  --label-source final-hash
```

Artifacts are written to `output/conditional_diffusion`: `condition_to_idx.json`,
`train_config.json`, `beta_schedule.json`, checkpoints, and sample PNGs.
MLX conditional training writes `config.json`, `label_map.json`, per-sample
source/final PNGs, `beta_schedule.json`, process traces, and
`checkpoints/step_*.{json,safetensors}` under its output directory. Samples are
saved under `sample/`, split into `source/` and `final/` subdirectories as
`source_*.png` and `final_*.png` with `source.labels.json` and
`final.labels.json` manifests. `sample/decode_comparison.json` records the
source/final RGB colors, Byte2RGB-decoded byte strings, and decoded-byte
hamming distances.
Use `--checkpoint-every N` to keep intermediate MLX checkpoints; the final MLX
checkpoint is always saved. MLX supports the same `linear`, `file`,
`hash-approach1`, and `hash-approach2` beta schedules as the PyTorch trainer.
For `linear`, `--timesteps auto` syncs the linear schedule length to the hash
approach schedule length.
Use `--save-train-batches-every N` to save actual training input batches as
PNG grids every `N` optimizer steps under `output/conditional_diffusion/train_batches`.
Each saved step also includes `step_XXXXXX.batch.json` with the exact source
image paths, labels, and conditions used in that batch.
When `--save-process-traces` is enabled, the forward process is saved for every
timestep and reverse process grids are saved under
`output/conditional_diffusion/process_traces`.

Training can be controlled with either `--train-steps` or `--epochs`.
When `--epochs` is set, the trainer uses
`ceil(dataset_size / batch_size) * epochs` optimizer updates.

`--condition-mode` is retained for backward compatibility, but training data
selection is fixed to `message.png`. `--label-source` only supports
`final-hash`; `Logs/4th Step` is no longer used as a base conditional DDPM
label.

Final-hash conditional model:
``` bash
python -m diffusion_hash_inv.models.conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_final_hash \
  --label-source final-hash \
  --image-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --timesteps 200 \
  --device auto
```

# Guided Conditional DDPM Training
Classifier guidance and classifier-free guidance are implemented separately in
`diffusion_hash_inv.models.guided_conditional_diffusion`, leaving the base
conditional DDPM module unchanged.
The notebook version is available at `notebooks/guided_conditional_diffusion.ipynb`.

Classifier-free guidance:
``` bash
python -m diffusion_hash_inv.models.guided_conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/guided_conditional_diffusion_cfg \
  --label-source final-hash \
  --guidance-mode classifier-free \
  --guidance-scale 2.0 \
  --condition-dropout 0.1 \
  --image-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --timesteps 200 \
  --device auto
```

Classifier guidance:
``` bash
python -m diffusion_hash_inv.models.guided_conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/guided_conditional_diffusion_classifier \
  --label-source final-hash \
  --guidance-mode classifier \
  --guidance-scale 1.0 \
  --classifier-base-channels 32 \
  --image-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --timesteps 200 \
  --device auto
```

Custom beta schedules can be used with:
``` bash
python -m diffusion_hash_inv.models.conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_custom_beta \
  --timesteps 200 \
  --beta-schedule hash-approach2
```
Supported `--beta-schedule` values are `linear`, `file`, `hash-approach1`,
and `hash-approach2`. For `file`, pass `--beta-values-path` pointing to a
JSON, TXT/CSV, NPY, or NPZ file containing beta values. With `file`,
`hash-approach1`, and `hash-approach2`, diffusion `timesteps` are inferred
from the resulting beta schedule length. For `linear`, `--timesteps` accepts
either an integer or `auto`; when set to `auto`, linear timesteps are synced
to the hash approach schedule length.

Process trace outputs:
```text
output/conditional_diffusion/process_traces/
  forward/
    x0.png
    t_000000.png
    ...
  reverse/
    xT_noise.png
    t_000199.png
    ...
    t_000000.png
```

## Command Line Argument
--hash_alg "Hash Algorithm": Hash algorithm  
-l "Length" /  -e "Exponential": Message Length  
-i: Iteration  
-m / -b: Setting Mode(message / bit)  
-v: Setting Verbose
-c: Clear data/output directory  

# SHA-256
## SHA-256 Properties
Message Size (bits): less than $2^{64}$  
Block Size (bits): $512 = 2^9$  
Word Size (bits): $32  = 2^5$
Message Digest Size (bits): 256  

# MD5
## MD5 Properties
