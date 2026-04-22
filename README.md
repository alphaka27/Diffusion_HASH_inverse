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