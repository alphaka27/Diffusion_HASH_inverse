# Method 1 — RGB Cube Binning

## Step 1 — 1D Axis Partitioning (Center Alignment)

Each RGB axis [0, 255] is split into **7 bins** using center alignment.

```
total_padding = 256 - (7 × 36) = 4  →  2 pixels of margin on each side
aligned range: [2, 253]

bin[0]: [  2,  37]
bin[1]: [ 38,  73]
bin[2]: [ 74, 109]
bin[3]: [110, 145]   ← center (index 3)
bin[4]: [146, 181]
bin[5]: [182, 217]
bin[6]: [218, 253]
```

> [0, 1] and [254, 255] are **intentional margins** — not covered by any bin.

---

## Step 2 — 3D Grid Construction

Cartesian product over $R \times G \times B$ axes $\rightarrow$ $7^3 = 343$ bins.  
Each bin is identified by `bin_idx = (r_idx, g_idx, b_idx)`.

---

## Step 3 — Exclusion Rules (343 → 256)

**Rule 1 — Boundary Exclusion**  
Exclude a bin if **2 or more** axes have a boundary index $\in \{0,\ 6\}$.

```
e.g. (0,0,*), (0,6,*), (6,0,*), (6,6,*), ...  →  68 bins excluded
```

**Rule 2 — Center Exclusion**  
Exclude a bin if **2 or more** axes have the center index $= 3$.

```
e.g. (3,3,*), (3,*,3), (*,3,3)  →  19 bins excluded
```

| | Count |
|---|---|
| Total 3D grid | 343 |
| Rule 1 exclusions | −68 |
| Rule 2 exclusions | −19 |
| **Valid bins** | **= 256** |

> Overlap between the two rules = 0 (boundary indices ≠ center index).

---

## Step 4 — Byte → Bin Assignment

The remaining 256 bins are assigned to **byte values `0x00`–`0xFF` in enumeration order**.

```
byte 0x00 → bin_idx=(0,1,1)  R:[  2, 37]  G:[ 38, 73]  B:[ 38, 73]
byte 0x01 → bin_idx=(0,1,2)  R:[  2, 37]  G:[ 38, 73]  B:[ 74,109]
byte 0x7F → bin_idx=(3,2,6)  R:[110,145]  G:[ 74,109]  B:[218,253]
byte 0xFF → bin_idx=(6,5,5)  R:[218,253]  G:[182,217]  B:[182,217]
```

---

## Step 5 — Encoding / Decoding

**Encoding**
```
byte  →  encoding_map[byte]  →  (R_range, G_range, B_range)
                              →  sample a random point via random.randint()
                              →  RGB(r, g, b)
```

**Decoding**
```
RGB(r,g,b)  →  iterate encoding_map
            →  find the bin whose R/G/B ranges all contain the point
               (inclusive check: start ≤ value ≤ end)
            →  return the matching key  (= original byte value)
```

> The same byte can produce **different RGB coordinates** on each call (reproducible via seed).  
> Decoding checks only **bin membership**, so any RGB point within a bin uniquely recovers the original byte.

---

## Design Intent

| Design Choice | Rationale |
|---|---|
| Margins [0,1] and [254,255] excluded | Buffer against boundary noise |
| Rule 1 — exclude bins with ≥2 boundary axes | Avoid extreme color combinations |
| Rule 2 — exclude bins with ≥2 center axes | Avoid over-concentration in grey-like region |
| Bin width = 36 | Largest width satisfying $7 \times 36 = 252 \leq 256$ |
| Final 256 bins | Guarantees **1-to-1 correspondence** with byte values 0–255 |


# Method 2 - RGB Cuboid Binning

## Step 1 - 1D Axis Partitioning
Split R, G axis into 8 bins  
Split B axis into 4 bins  

## Step 2 - 3D Chunk Consturction
