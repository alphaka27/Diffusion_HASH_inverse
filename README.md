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
to an equal-area square (e.g. `7168x28 -> 448x448`). `height-flatten` uses the
`ImgConfig.img_size` height as the unit, flattens ImgConfig-sized blocks, and
then reshapes to a square RGB image. The source dimensions must be multiples of
`ImgConfig.img_size` (`28x28` by default). `pad` and `resize` are also available.

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

Artifacts are written to `output/conditional_diffusion`: `condition_to_idx.json`,
`train_config.json`, `beta_schedule.json`, checkpoints, and sample grids.
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
