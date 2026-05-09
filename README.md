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
