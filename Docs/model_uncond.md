# Model 5 — Unconditional DDPM (`uncond`)

> 구현: `src/diffusion_hash_inv/models/unconditional_ddpm.py`  
> 노트북 job 접미사: `_uncond_linear`, `_uncond_approach1`, `_uncond_approach2`

---

## 개요

| 항목 | 값 |
|------|-----|
| 모델 클래스 | `UnconditionalUNet` |
| 스케줄러 | `UnconditionalDDPMScheduler` |
| 모델 수 | 1개 |
| 조건 | **없음** |
| label / hash | 사용 안 함 |
| guidance scale | 없음 |

---

## 아키텍처 — UnconditionalUNet

Conditional DDPM과 동일한 U-Net 골격을 사용하되, **condition embedding이 없고** timestep embedding 만 사용한다.

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

emb = e_t(t)   ← timestep embedding만 사용
```

```python
# UnconditionalUNet.forward()
emb = self.time_embedding(timesteps)   # condition embedding 없음
...
```

---

## 학습 과정

### Forward process

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I)
$$

### Noise prediction

label 없이 이미지와 timestep만 입력한다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t)
$$

### Loss

$$
\mathcal{L}_{uncond}
= \mathbb{E}_{x_0,\,t,\,\epsilon}
\left[\|\epsilon - \epsilon_\theta(x_t,\, t)\|_2^2\right]
$$

모델은 특정 label 을 구분하지 않고 **전체 데이터 분포의 marginal** 을 학습한다.

$$
p_\theta(x_0) = \sum_y p(y)\,p_\theta(x_0 \mid y)
$$

---

## 샘플링 과정

$x_T \sim \mathcal{N}(0,I)$ 에서 시작해 $t = T-1,\ldots,0$ 반복:

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t)
$$

$$
\mu_\theta
= \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}\right)
$$

$$
x_{t-1} =
\begin{cases}
\mu_\theta + \sqrt{\tilde{\beta}_t}\,z, & t > 0,\quad z\sim\mathcal{N}(0,I) \\
\mu_\theta, & t = 0
\end{cases}
$$

샘플링 시 어떤 label 로도 유도되지 않으므로, 결과는 학습 데이터 전반의 평균 특성을 반영한다.

---

## 다른 모델과의 비교

| | **uncond** | **base** | **guided_cfg** |
|--|:----------:|:--------:|:--------------:|
| 조건 | 없음 | label $y$ | label $y$ + null $\varnothing$ |
| embedding | $e_t(t)$ 만 | $e_t + e_c$ | $e_t + e_c$ (null 포함) |
| 학습 분포 | $p(x_0)$ (marginal) | $p(x_0 \mid y)$ | $p(x_0 \mid y)$ + $p(x_0)$ |
| 용도 | baseline | 조건부 생성 | 강화된 조건부 생성 |

---

## 프로젝트에서의 역할

Unconditional DDPM 은 다른 조건부 모델들의 **비교 baseline** 으로 사용된다.

- 조건 신호가 없을 때 생성 품질이 어느 수준인지 확인
- label 정보가 생성에 얼마나 기여하는지 측정하는 기준점

---

## 흐름 요약

```
[학습]
x0  →  q_sample(x0, t)  →  x_t
                              │
                       UNet(x_t, t) → ε̂      ← label 없음
                              │
                       MSE(ε, ε̂) → backprop

[샘플링]
x_T ~ N(0,I)
  for t = T-1 … 0:
    ε̂ = UNet(x_t, t)            ← label 없음
    x_{t-1} = posterior_mean(x_t, ε̂) + noise
```
