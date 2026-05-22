# Model 4 — Loop-Conditioned DDPM (`loop`)

> 구현: `src/diffusion_hash_inv/models/loop_conditioned_diffusion.py`  
> 노트북 job 접미사: `_loop_linear`, `_loop_approach1`, `_loop_approach2`

---

## 개요

| 항목 | 값 |
|------|-----|
| 모델 클래스 | `LoopConditionedUNet` |
| 스케줄러 | `LoopConditionedDDPMScheduler` |
| 모델 수 | 1개 |
| 조건 | MD5 Step-4 loop-state tensor $S$ |
| condition shape | $(L+2) \times 4 = 66 \times 4$ |

---

## 조건 — Loop-State Tensor $S$

MD5 Step 4 (`4th Step / 1st Round`) 의 각 loop 단계에서 내부 레지스터 상태를 수집한다.

$$
S \in [-1,\,1]^{(L+2)\times 4},
\qquad L = 64
$$

각 row 는 MD5 A, B, C, D 레지스터 4개의 상태다.

$$
S_k = [A_k,\; B_k,\; C_k,\; D_k],
\qquad k = 0,\ldots,L+1
$$

loop start / end 경계 상태를 포함해 총 $L+2 = 66$ rows다.

### 정규화

uint32 word 를 $[-1, 1]$ 범위로 정규화한다.

$$
\mathrm{norm}(v) = \frac{v}{2^{32}-1} \cdot 2 - 1
$$

---

## 아키텍처 — LoopConditionedUNet

Conditional DDPM과 동일한 U-Net 골격을 사용하되, condition embedding이 3개 신호의 합으로 구성된다.

```
입력 x  (C × H × W)
  │
  ├─ input Conv(C → B)
  ├─ down1  ConditionalResBlock(B,  B,   emb)
  ├─ down   Conv(B  → 2B, stride=2)
  ├─ down2  ConditionalResBlock(2B, 2B,  emb)
  ├─ down   Conv(2B → 4B, stride=2)
  ├─ mid    ConditionalResBlock(4B, 4B,  emb)
  ├─ up     ConvT(4B → 2B) + skip
  ├─ up1    ConditionalResBlock(4B, 2B,  emb)
  ├─ up     ConvT(2B → B)  + skip
  ├─ up2    ConditionalResBlock(2B, B,   emb)
  └─ output GroupNorm → SiLU → Conv(B → C)

emb = e_t(t) + e_s(S_{k(t)}) + e_p(k(t))
```

---

## Timestep → Loop-State 매핑

Diffusion timestep $t$ 를 loop state index $k(t)$ 로 변환한다.

$$
k(t)
= \left\lfloor \frac{t \cdot (L+2)}{T} \right\rfloor
$$

각 diffusion step 이 서로 다른 loop 단계에 대응되어, denoising 방향이 MD5 연산 흐름과 정렬된다.

---

## Conditioning 벡터 $\mathbf{e}$

세 가지 embedding 의 합으로 conditioning 벡터를 구성한다.

$$
\mathbf{e}
= e_t(t) + e_s(S_{k(t)}) + e_p(k(t))
$$

| 항목 | 수식 | 구현 |
|------|------|------|
| Timestep embedding | $e_t(t) = \mathrm{MLP}(\mathrm{SinusoidalPE}(t))$ | `time_embedding` |
| State value embedding | $e_s(S_{k(t)}) = \mathrm{MLP}(S_{k(t)})$ | `state_embedding` (Linear → SiLU → Linear) |
| State position embedding | $e_p(k(t)) = \mathrm{MLP}(\mathrm{SinusoidalPE}(k(t)))$ | `state_position_embedding` |

```python
# LoopConditionedUNet._embedding()
k = timestep_to_state_indices(timesteps, state_count=66, ...)
selected_states = conditions[batch_idx, k]           # S_{k(t)}
e = time_embedding(t)
  + state_embedding(selected_states)                 # e_s
  + state_position_embedding(k)                      # e_p
```

---

## 학습 과정

### Forward process

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I)
$$

### Noise prediction

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, S)
$$

### Loss

$$
\mathcal{L}_{loop}
= \mathbb{E}_{x_0,\,t,\,S,\,\epsilon}
\left[\|\epsilon - \epsilon_\theta(x_t, t, S)\|_2^2\right]
$$

---

## 샘플링 과정

$x_T \sim \mathcal{N}(0,I)$ 에서 시작해 $t = T-1,\ldots,0$ 반복:

$$
k(t) = \left\lfloor \frac{t\cdot(L+2)}{T} \right\rfloor
$$

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, S)
$$

$$
\mu_\theta
= \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}\right)
$$

$$
x_{t-1} =
\begin{cases}
\mu_\theta + \sqrt{\tilde{\beta}_t}\,z, & t > 0 \\
\mu_\theta, & t = 0
\end{cases}
$$

---

## Conditional DDPM과의 차이

| | **base (Conditional)** | **loop** |
|--|:----------------------:|:--------:|
| 조건 타입 | scalar label $y$ | matrix $S \in \mathbb{R}^{66\times4}$ |
| 조건 embedding | `nn.Embedding` | MLP(state) + SinusoidalPE(position) |
| embedding 구성 | $e_t + e_c$ | $e_t + e_s + e_p$ |
| Timestep-Condition 연계 | 없음 | $k(t)$ 매핑으로 연계됨 |

---

## 흐름 요약

```
[학습]
x0, S  →  q_sample(x0, t)  →  x_t
                                 │
              k(t) = floor(t*(L+2)/T)
              e = e_t(t) + e_s(S_{k(t)}) + e_p(k(t))
                                 │
                          UNet(x_t, t, S) → ε̂
                                 │
                          MSE(ε, ε̂) → backprop

[샘플링]
x_T ~ N(0,I)
  for t = T-1 … 0:
    k(t) = floor(t*(L+2)/T)
    ε̂ = UNet(x_t, t, S)   ← S_{k(t)} 내부 선택
    x_{t-1} = posterior_mean(x_t, ε̂) + noise
```
