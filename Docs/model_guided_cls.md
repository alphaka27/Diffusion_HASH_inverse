# Model 3 — Guided DDPM · Classifier Guidance (`guided_cls`)

> 구현: `src/diffusion_hash_inv/models/guided_conditional_diffusion.py`  
> 노트북 job 접미사: `_guided_cls_linear`, `_guided_cls_approach1`, `_guided_cls_approach2`

---

## 개요

| 항목 | 값 |
|------|-----|
| 모델 클래스 | `ConditionalUNet` + `NoisyImageClassifier` |
| 스케줄러 | `DDPMNoiseScheduler` |
| 모델 수 | **2개** |
| Denoiser 조건 | 고정 label `0` (단일 class) |
| Classifier 조건 | 실제 label $y$ |
| `guidance_scale` $w$ | **1.0** |
| `condition_dropout` | 0.0 |

---

## 아키텍처

### Denoiser — ConditionalUNet

Conditional DDPM과 동일한 U-Net 구조이나, `num_conditions = 1` 로 고정되어  
label이 아닌 **단일 더미 class** 를 입력받는다 (사실상 unconditional denoiser).

### Classifier — NoisyImageClassifier

noisy 이미지 $x_t$ 와 timestep $t$ 를 입력받아 label $y$ 의 확률을 예측하는 작은 CNN.

```
입력 x_t  (C × H × W)
  │
  ├─ input  Conv(C → B_c)
  ├─        + time_projection(e_t(t))   ← additive bias
  ├─ block1 GroupNorm → SiLU → Conv(B_c → B_c)
  ├─ down1  GroupNorm → SiLU → Conv(B_c → 2B_c, stride=2)
  ├─ down2  GroupNorm → SiLU → Conv(2B_c → 4B_c, stride=2)
  └─ head   GroupNorm → SiLU → AdaptiveAvgPool → Linear(4B_c → N)

B_c = classifier_base_channels (full: 64)
```

---

## 학습 과정

두 모델을 **독립적으로** 각각 학습한다.

### Denoiser 학습

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

label 자리에 `0` 을 고정해 사실상 unconditional denoiser 로 학습한다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, 0)
$$

$$
\mathcal{L}_{denoiser}
= \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t, 0)\|_2^2\right]
$$

### Classifier 학습

동일한 $x_t$ 에서 label $y$ 를 예측한다.

$$
p_\phi(y \mid x_t, t) = \mathrm{softmax}\!\left(f_\phi(x_t, t)\right)_y
$$

$$
\mathcal{L}_{cls} = -\log p_\phi(y \mid x_t, t)
$$

---

## 샘플링 과정

### Step 1 — Denoiser로 posterior mean 계산

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, 0)
$$

$$
\mu_\theta
= \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}\right)
$$

### Step 2 — Classifier gradient 계산

$$
g = \nabla_{x_t} \log p_\phi(y \mid x_t,\, t)
$$

```python
# _classifier_log_prob_gradient()
x_in = x.detach().requires_grad_(True)
logits = classifier(x_in, timesteps)
selected = log_softmax(logits).gather(1, labels).sum()
grad = autograd.grad(selected, x_in)[0]
```

### Step 3 — Guided mean 계산

$$
\boxed{
\mu_{guided}
= \mu_\theta(x_t,\, t,\, 0)
+ \tilde{\beta}_t\, w\, g
}
$$

posterior variance $\tilde{\beta}_t$ 로 스케일된 gradient 를 평균에 더한다.

### Step 4 — Reverse step

$$
x_{t-1} =
\begin{cases}
\mu_{guided} + \sqrt{\tilde{\beta}_t}\,z, & t > 0,\quad z\sim\mathcal{N}(0,I) \\
\mu_{guided}, & t = 0
\end{cases}
$$

---

## CFG와의 비교

| | **guided_cfg** | **guided_cls** |
|--|:--------------:|:--------------:|
| 추가 모델 | 없음 | NoisyImageClassifier |
| 조건 신호 | null label 보간 | $\nabla \log p(y\mid x_t)$ |
| 샘플링 forward | UNet ×2 | UNet ×1 + Classifier gradient |
| 학습 데이터 요구 | 단일 모델 | 두 모델 별도 학습 |

---

## 흐름 요약

```
[학습]
x0, y  →  q_sample(x0, t)  →  x_t
  ├─ Denoiser:  UNet(x_t, t, 0)   → ε̂  →  MSE loss
  └─ Classifier: CNN(x_t, t)       → logits → CE loss(y)

[샘플링]
x_T ~ N(0,I)
  for t = T-1 … 0:
    ε̂        = UNet(x_t, t, 0)
    μ_θ       = posterior_mean(x_t, ε̂)
    g         = ∇_{x_t} log p_φ(y | x_t, t)
    μ_guided  = μ_θ + β̃_t * w * g
    x_{t-1}   = μ_guided + √β̃_t * z
```
