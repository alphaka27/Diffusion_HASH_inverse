# Model 1 — Conditional DDPM (`base`)

> 구현: `src/diffusion_hash_inv/models/conditional_diffusion.py`  
> 노트북 job 접미사: `_base_linear`, `_base_approach1`, `_base_approach2`

---

## 개요

| 항목 | 값 |
|------|-----|
| 모델 클래스 | `ConditionalUNet` |
| 스케줄러 | `DDPMNoiseScheduler` |
| 모델 수 | 1개 |
| 조건 | final hash label $y \in \{0,\ldots,N-1\}$ |
| `num_conditions` | $N$ |
| guidance scale | 없음 |
| label dropout | 없음 |

---

## 아키텍처 — ConditionalUNet

```
입력 x  (C × H × W)
  │
  ├─ input Conv(C → B)
  ├─ down1  ConditionalResBlock(B,  B,   emb)
  ├─ down   Conv(B  → 2B, stride=2)
  ├─ down2  ConditionalResBlock(2B, 2B,  emb)
  ├─ down   Conv(2B → 4B, stride=2)
  ├─ mid    ConditionalResBlock(4B, 4B,  emb)
  ├─ up     ConvT(4B → 2B)  + skip(down2)
  ├─ up1    ConditionalResBlock(4B, 2B,  emb)
  ├─ up     ConvT(2B → B)   + skip(down1)
  ├─ up2    ConditionalResBlock(2B, B,   emb)
  └─ output GroupNorm → SiLU → Conv(B → C)

B = base_channels (full: 64)
emb = e_t(t) + e_c(y)  ← 모든 ResBlock에 공유
```

### Conditioning 벡터 $\mathbf{e}$

$$
e_t(t) = \mathrm{MLP}\!\left(\mathrm{SinusoidalPE}(t)\right) \in \mathbb{R}^d
$$

$$
e_c(y) = W_c\, \mathbf{1}_y \in \mathbb{R}^d \qquad \text{(nn.Embedding)}
$$

$$
\mathbf{e} = e_t(t) + e_c(y) \in \mathbb{R}^d
$$

### ConditionalResBlock 내 주입

$$
h^{(1)} = \mathrm{Conv}_1\!\left(\sigma\!\left(\mathrm{Norm}(x)\right)\right)
$$

$$
h^{(2)} = h^{(1)} + W_e\,\sigma(\mathbf{e})\cdot\mathbf{1}_{H\times W}
$$

$$
h^{(3)} = \mathrm{Conv}_2\!\left(\sigma\!\left(\mathrm{Norm}(h^{(2)})\right)\right) + \mathrm{skip}(x)
$$

---

## 학습 과정

### Forward process

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I)
$$

### Noise prediction

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, y)
$$

### Loss

$$
\mathcal{L}
= \mathbb{E}_{x_0,\,t,\,y,\,\epsilon}
\left[\|\epsilon - \epsilon_\theta(x_t,t,y)\|_2^2\right]
$$

---

## 샘플링 과정

$x_T \sim \mathcal{N}(0,I)$ 에서 시작해 $t = T-1,\ldots,0$ 반복:

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, y)
$$

$$
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}\right)
$$

$$
x_{t-1} =
\begin{cases}
\mu_\theta + \sqrt{\tilde{\beta}_t}\,z, & t > 0,\quad z\sim\mathcal{N}(0,I) \\
\mu_\theta, & t = 0
\end{cases}
$$

---

## 흐름 요약

```
[학습]
x0, y  →  q_sample(x0, t)  →  x_t
                                 │
                          UNet(x_t, t, y) → ε̂
                                 │
                          MSE(ε, ε̂) → backprop

[샘플링]
x_T ~ N(0,I)
  for t = T-1 … 0:
    ε̂ = UNet(x_t, t, y)
    x_{t-1} = posterior_mean(x_t, ε̂) + noise
```
