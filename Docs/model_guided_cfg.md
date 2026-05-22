# Model 2 — Guided DDPM · Classifier-Free Guidance (`guided_cfg`)

> 구현: `src/diffusion_hash_inv/models/guided_conditional_diffusion.py`  
> 노트북 job 접미사: `_guided_cfg_linear`, `_guided_cfg_approach1`, `_guided_cfg_approach2`

---

## 개요

| 항목 | 값 |
|------|-----|
| 모델 클래스 | `ConditionalUNet` |
| 스케줄러 | `DDPMNoiseScheduler` |
| 모델 수 | **1개** |
| 조건 | label $y$ 또는 null label $\varnothing$ |
| `num_conditions` | $N + 1$ (null 포함) |
| `guidance_scale` $w$ | **2.0** |
| `condition_dropout` $p_{drop}$ | **0.1** |

---

## 아키텍처

Conditional DDPM과 동일한 `ConditionalUNet` 구조를 사용하되,  
null label $\varnothing$ 을 추가 class로 포함한다.

$$
\text{num\_conditions} = N + 1,
\qquad
\varnothing \equiv N \;\text{(null index)}
$$

---

## 학습 과정

### Condition Dropout

매 step마다 확률 $p_{drop}$ 으로 실제 label을 null label로 치환한다.

$$
c' =
\begin{cases}
y & \text{with probability } 1 - p_{drop} \\
\varnothing & \text{with probability } p_{drop}
\end{cases}
$$

```python
labels = apply_condition_dropout(labels, null_label=N, dropout=0.1)
```

### Forward process

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I)
$$

### Noise prediction & Loss

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, c')
$$

$$
\mathcal{L}_{cfg}
= \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t, c')\|_2^2\right]
$$

단일 모델이 조건부($c' = y$)와 무조건부($c' = \varnothing$) noise 예측을 **함께** 학습한다.

---

## 샘플링 과정

동일 모델을 **두 번 forward** 해 예측을 보간한다.

### Step 1 — 조건부·무조건부 noise 예측

$$
\epsilon_{uncond} = \epsilon_\theta(x_t,\, t,\, \varnothing)
$$

$$
\epsilon_{cond}   = \epsilon_\theta(x_t,\, t,\, y)
$$

### Step 2 — CFG 보간

$$
\boxed{
\epsilon_{cfg}
= \epsilon_{uncond} + w\,(\epsilon_{cond} - \epsilon_{uncond})
}
$$

$w = 2.0$ 이면 조건부 방향을 오버슈팅(extrapolation)한다.

$$
= -\epsilon_{uncond} + 2\,\epsilon_{cond}
$$

### Step 3 — Reverse step

$$
\mu_{cfg}
= \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_{cfg}\right)
$$

$$
x_{t-1} =
\begin{cases}
\mu_{cfg} + \sqrt{\tilde{\beta}_t}\,z, & t > 0 \\
\mu_{cfg}, & t = 0
\end{cases}
$$

---

## $w$ 값에 따른 동작

| $w$ | $\epsilon_{cfg}$ | 동작 |
|-----|-------------------|------|
| 0.0 | $\epsilon_{uncond}$ | 완전 unconditional |
| 1.0 | $\epsilon_{cond}$ | 순수 conditional (base 모델과 수식 동일) |
| **2.0** | $-\epsilon_{uncond} + 2\epsilon_{cond}$ | **조건 방향 extrapolation (현재 설정)** |

---

## 흐름 요약

```
[학습]
x0, y  →  dropout(y, p=0.1) → c'
       →  q_sample(x0, t)   → x_t
                                │
                         UNet(x_t, t, c') → ε̂
                                │
                         MSE(ε, ε̂) → backprop

[샘플링]
x_T ~ N(0,I)
  for t = T-1 … 0:
    ε_uncond = UNet(x_t, t, ∅)
    ε_cond   = UNet(x_t, t, y)
    ε_cfg    = ε_uncond + 2.0 * (ε_cond - ε_uncond)
    x_{t-1}  = posterior_mean(x_t, ε_cfg) + noise
```
