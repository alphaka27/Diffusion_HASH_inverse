# 노트북 기준 5개 Diffusion 모델 동작 정리

> `ddpm_torch_mlx_equivalent_usage.ipynb` 에서 사용하는 모델 변형을 기준으로 한다.
> 공통 noise schedule 수식은 `DiffusionModels.md` 를 참조한다.

---

## 공통 기반 수식

### Forward process (q-sample)

학습 이미지 $x_0$ 에 임의 noise $\epsilon$ 을 섞어 $x_t$ 를 만든다.

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon
$$

### Reverse step (p-sample)

모델의 noise 예측값 $\hat{\epsilon}$ 으로 posterior mean 을 계산한다.

$$
\mu_\theta(x_t, t, c)
= \frac{1}{\sqrt{\alpha_t}}
  \left(
    x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \hat{\epsilon}
  \right)
$$

$t > 0$ 이면 posterior variance $\tilde{\beta}_t$ 를 더해 다음 sample 을 만든다.

$$
x_{t-1} = \mu_\theta(x_t, t, c) + \sqrt{\tilde{\beta}_t}\, z,
\qquad z \sim \mathcal{N}(0, I)
$$

$t = 0$ 에서는 noise 를 더하지 않는다.

$$
x_0 = \mu_\theta(x_0, t=0, c)
$$

### 학습 Loss

모든 모델의 기본 loss 는 noise prediction MSE 이다.

$$
\mathcal{L}
= \mathbb{E}_{x_0,\,t,\,\epsilon}
  \left[\, \|\epsilon - \hat{\epsilon}\|_2^2 \,\right]
$$

---

## 모델 1 — Conditional DDPM (`base`)

### 개요

| 항목 | 값 |
|------|-----|
| 모듈 | `conditional_diffusion.py` |
| 모델 수 | 1개 (`ConditionalUNet`) |
| 조건 | final hash label $y$ |
| `num_conditions` | $N$ (실제 클래스 수) |
| guidance scale | 없음 |

### 동작 과정

**학습**
1. 이미지 $x_0$ 와 label $y$ 를 로드한다.
2. $t \sim \text{Uniform}(0, T-1)$ 을 샘플링하고 $x_t$ 를 생성한다.
3. U-Net 에 $(x_t,\, t,\, y)$ 를 입력해 noise 를 예측한다.
4. MSE loss 로 역전파한다.

**샘플링**
1. $x_T \sim \mathcal{N}(0, I)$ 로 시작한다.
2. $t = T-1, \ldots, 0$ 까지 아래를 반복한다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, y)
$$

$$
x_{t-1} = \mu_\theta(x_t, t, y) + \sqrt{\tilde{\beta}_t}\, z
$$

### 수식 요약

| 단계 | 수식 |
|------|------|
| Condition embedding | $e_c = \mathrm{Embedding}(y)$ |
| UNet 입력 | $h(t, y) = e_t(t) + e_c(y)$ |
| Noise prediction | $\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, y)$ |
| Loss | $\mathcal{L} = \|\epsilon - \hat{\epsilon}\|_2^2$ |

---

## 모델 2 — Guided DDPM · Classifier-Free Guidance (`guided_cfg`)

### 개요

| 항목 | 값 |
|------|-----|
| 모듈 | `guided_conditional_diffusion.py` |
| 모델 수 | 1개 (`ConditionalUNet`) |
| 조건 | final hash label $y$ 또는 null label $\varnothing$ |
| `num_conditions` | $N + 1$ (null 포함) |
| `guidance_scale` $w$ | 2.0 |
| `condition_dropout` $p_{drop}$ | 0.1 |

### 동작 과정

**학습**

확률 $p_{drop}$ 으로 label 을 null label 로 치환한다.

$$
c' =
\begin{cases}
y & \text{with probability } 1 - p_{drop} \\
\varnothing & \text{with probability } p_{drop}
\end{cases}
$$

치환된 $c'$ 로 noise 예측 및 MSE loss 를 계산한다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, c')
$$

$$
\mathcal{L}_{cfg} = \|\epsilon - \hat{\epsilon}\|_2^2
$$

**샘플링**

동일 모델을 두 번 forward 해 조건부·무조건부 예측을 보간한다.

$$
\epsilon_{uncond} = \epsilon_\theta(x_t,\, t,\, \varnothing)
$$

$$
\epsilon_{cond}   = \epsilon_\theta(x_t,\, t,\, y)
$$

$$
\boxed{
\epsilon_{cfg}
= \epsilon_{uncond} + w\,(\epsilon_{cond} - \epsilon_{uncond})
}
$$

$w = 2.0$ 이면 조건부 방향을 오버슈팅(extrapolation)한다.

$$
\mu_{cfg}
= \frac{1}{\sqrt{\alpha_t}}
  \left(
    x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_{cfg}
  \right)
$$

### 특수 케이스

| $w$ | 동작 |
|-----|------|
| 0.0 | 순수 unconditional ($\epsilon_{uncond}$) |
| 1.0 | 순수 conditional ($\epsilon_{cond}$) — 수식상 base 모델과 동일 |
| 2.0 | 조건 방향 extrapolation (현재 설정) |

---

## 모델 3 — Guided DDPM · Classifier Guidance (`guided_cls`)

### 개요

| 항목 | 값 |
|------|-----|
| 모듈 | `guided_conditional_diffusion.py` |
| 모델 수 | **2개** (`ConditionalUNet` + `NoisyImageClassifier`) |
| `guidance_scale` $w$ | 1.0 |
| `condition_dropout` | 0.0 |

### 동작 과정

**학습 — Denoiser**

Denoiser 는 label `0` 고정(unconditional에 준하는 단일 class) 으로 학습한다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, 0)
$$

$$
\mathcal{L}_{denoiser} = \|\epsilon - \hat{\epsilon}\|_2^2
$$

**학습 — Classifier**

`NoisyImageClassifier` 는 noisy 이미지 $x_t$ 에서 label $y$ 를 예측한다.

$$
p_\phi(y \mid x_t, t) = \mathrm{softmax}(f_\phi(x_t, t))_y
$$

$$
\mathcal{L}_{cls} = -\log p_\phi(y \mid x_t, t)
$$

**샘플링**

Denoiser 로 posterior mean 을 계산한 뒤, classifier gradient 로 mean 을 보정한다.

$$
g = \nabla_{x_t} \log p_\phi(y \mid x_t, t)
$$

$$
\boxed{
\mu_{guided}
= \mu_\theta(x_t, t, 0) + \tilde{\beta}_t\, w\, g
}
$$

$$
x_{t-1} = \mu_{guided} + \sqrt{\tilde{\beta}_t}\, z
$$

### 수식 요약

| 단계 | 수식 |
|------|------|
| Denoiser loss | $\|\epsilon - \epsilon_\theta(x_t, t, 0)\|_2^2$ |
| Classifier loss | $-\log p_\phi(y \mid x_t, t)$ |
| Guidance gradient | $g = \nabla_{x_t} \log p_\phi(y \mid x_t, t)$ |
| Guided mean | $\mu_{guided} = \mu_\theta + \tilde{\beta}_t\, w\, g$ |

---

## 모델 4 — Loop-Conditioned DDPM (`loop`)

### 개요

| 항목 | 값 |
|------|-----|
| 모듈 | `loop_conditioned_diffusion.py` |
| 모델 수 | 1개 (`LoopConditionedUNet`) |
| 조건 | MD5 Step-4 loop-state tensor $S$ |
| Condition shape | $S \in [-1, 1]^{(L+2) \times 4}$, $L = 64$ |

### Loop-state Tensor

각 row 는 MD5 내부 상태 벡터이다.

$$
S_k = [A_k,\, B_k,\, C_k,\, D_k], \qquad k = 0, \ldots, L+1
$$

uint32 word 는 다음으로 정규화한다.

$$
\mathrm{norm}(v) = \frac{v}{2^{32} - 1} \cdot 2 - 1
$$

### Timestep → Loop-state 매핑

Diffusion timestep $t$ 를 loop state index $k(t)$ 로 변환한다.

$$
k(t) = \left\lfloor \frac{t \cdot (L+2)}{T} \right\rfloor
$$

### Condition Embedding

선택된 loop state $S_{k(t)}$ 는 세 가지 embedding 의 합으로 U-Net 에 주입된다.

$$
h(t, S)
= e_t(t)
+ e_s(S_{k(t)})
+ e_p(k(t))
$$

| 기호 | 의미 |
|------|------|
| $e_t(t)$ | sinusoidal timestep embedding |
| $e_s(S_{k(t)})$ | loop state value linear projection |
| $e_p(k(t))$ | state position (index) sinusoidal embedding |

### 수식 요약

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, S)
$$

$$
\mathcal{L}_{loop} = \|\epsilon - \hat{\epsilon}\|_2^2
$$

Reverse sampling 은 공통 p-sample 수식과 동일하며, condition $c$ 자리에 $S$ 를 사용한다.

---

## 모델 5 — Unconditional DDPM (`uncond`)

### 개요

| 항목 | 값 |
|------|-----|
| 모듈 | `unconditional_ddpm.py` |
| 모델 수 | 1개 (`UnconditionalUNet`) |
| 조건 | 없음 |

### 동작 과정

**학습**

Label 을 사용하지 않고 이미지 분포만 학습한다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t)
$$

$$
\mathcal{L}_{uncond} = \|\epsilon - \hat{\epsilon}\|_2^2
$$

**샘플링**

$x_T \sim \mathcal{N}(0, I)$ 에서 시작해 $t = T-1, \ldots, 0$ 까지 반복한다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t)
$$

$$
x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\tilde{\beta}_t}\, z
$$

데이터 분포만을 학습하는 baseline 으로, label 조건 없이 이미지를 생성한다.

---

## 5개 모델 비교 요약

| | **base** | **guided_cfg** | **guided_cls** | **loop** | **uncond** |
|--|:--------:|:--------------:|:--------------:|:--------:|:----------:|
| 모델 수 | 1 | 1 | **2** | 1 | 1 |
| 조건 종류 | label $y$ | label $y$ + null $\varnothing$ | label $y$ | loop state $S$ | 없음 |
| `num_conditions` | $N$ | $N+1$ | 1 (fixed `0`) | — | — |
| 학습 시 label dropout | ❌ | ✅ ($p=0.1$) | ❌ | ❌ | ❌ |
| 샘플링 forward 횟수 | 1 | **2** | 1 + gradient | 1 | 1 |
| `guidance_scale` | — | 2.0 | 1.0 | — | — |
| Classifier 학습 | ❌ | ❌ | ✅ | ❌ | ❌ |
| 조건 주입 방식 | label embedding | label/null embedding | classifier gradient | loop-state embedding | 없음 |
