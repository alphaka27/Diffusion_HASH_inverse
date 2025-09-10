# Diffusion HASH inverse
Finding the hash inverse using a diffusion model

# SHA-256
## SHA-256 Properties
Message Size (bits): less than $2^{64}$  
Block Size (bits): $512 = 2^9$  
Word Size (bits): $32  = 2^5$
Message Digest Size (bits): 256  


```
Diffusion_HASH_inverse
├─ .python-version
├─ README.md
├─ data
│  └─ binary
│     ├─ random_1024_bits.bin
│     ├─ random_32_bits.bin
│     ├─ random_4096_bits.bin
│     └─ random_512_bits.bin
├─ notebooks
│  └─ sha_256.ipynb
├─ output
├─ pyproject.toml
├─ src
│  ├─ diffusion_hash_inv
│  │  ├─ __init__.py
│  │  ├─ generator
│  │  │  ├─ Password_rule.md
│  │  │  ├─ __init__.py
│  │  │  ├─ nist_pwgen_utf8.py
│  │  │  ├─ random_n_bits.py
│  │  │  └─ random_n_char.py
│  │  ├─ hashing
│  │  │  ├─ __init__.py
│  │  │  └─ sha_256.py
│  │  └─ utils
│  │     ├─ __init__.py
│  │     ├─ file_io.py
│  │     └─ project_root.py
│  └─ diffusion_test.py
└─ tests
   └─ test.py

```