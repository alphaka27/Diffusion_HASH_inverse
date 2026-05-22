# Conditional Diffusion Model에서 Condition의 역할

> 구현 기준: `src/diffusion_hash_inv/models/conditional_diffusion.py`  
> 대상 클래스: `ConditionalUNet`, `ConditionalResBlock`

---

## 1. 한 줄 요약

> **Condition은 "어떤 이미지를 생성할지"를 U-Net의 모든 residual block에 알려주는 신호다.**

---

## 2. Condition Embedding

label $y \in \{0, \ldots, N-1\}$ 를 $d$-차원 벡터로 변환한다.

$$
e_c(y) = W_c\, \mathbf{1}_y \in \mathbb{R}^d
\qquad \text{(nn.Embedding)}
$$

$W_c \in \mathbb{R}^{N \times d}$ 는 학습 가능한 embedding 행렬이며,  
$\mathbf{1}_y$ 는 index $y$ 에 해당하는 one-hot 선택이다.

---

## 3. Time + Condition 합산

Sinusoidal timestep embedding $e_t(t)$ 와 합산해 하나의 conditioning 벡터를 만든다.

$$
\mathbf{e} = e_t(t) + e_c(y) \in \mathbb{R}^d
$$

두 신호가 **동일한 벡터 공간에서 더해지므로**, condition은 timestep과 동등한 위상으로 네트워크에 전달된다.

구현:

```python
# ConditionalUNet._embedding()
time_emb = self.time_embedding(timesteps)   # sinusoidal → MLP → R^d
cond_emb = self.condition_embedding(labels) # nn.Embedding → R^d
return time_emb + cond_emb                  # e = e_t(t) + e_c(y)
```

---

## 4. ResBlock 내 주입 — Additive Feature Modulation

각 `ConditionalResBlock` 에서 $\mathbf{e}$ 는 feature map $h$ 에 **additive bias** 형태로 공간 전체에 broadcast된다.

$$
h^{(1)} = \mathrm{Conv}_1\!\left(\sigma\!\left(\mathrm{Norm}(x)\right)\right)
$$

$$
\boxed{
h^{(2)} = h^{(1)} + W_e\,\sigma(\mathbf{e}) \cdot \mathbf{1}_{H \times W}
}
$$

$$
h^{(3)} = \mathrm{Conv}_2\!\left(\sigma\!\left(\mathrm{Norm}(h^{(2)})\right)\right) + \mathrm{skip}(x)
$$

| 기호 | 의미 |
|------|------|
| $W_e \in \mathbb{R}^{C_{out} \times d}$ | `emb_proj` (Linear) |
| $\sigma$ | SiLU 활성화 함수 |
| $\mathbf{1}_{H \times W}$ | 공간 차원 broadcast `[:, :, None, None]` |

구현:

```python
# ConditionalResBlock.forward()
h = self.conv1(F.silu(self.norm1(x)))
h = h + self.emb_proj(F.silu(emb))[:, :, None, None]  # additive bias broadcast
h = self.conv2(F.silu(self.norm2(h)))
return h + self.skip(x)
```

---

## 5. 전체 U-Net Forward

공유된 $\mathbf{e}$ 가 5개 블록(down1, down2, mid, up1, up2) 모두에 주입된다.

$$
\mathbf{e} = e_t(t) + e_c(y)
$$

$$
\hat{\epsilon} = f_{\mathrm{out}}\!\Bigl(
  f_{\mathrm{up2}}\bigl(\cdots,\, \mathbf{e}\bigr),\;\;
  f_{\mathrm{up1}}\bigl(\cdots,\, \mathbf{e}\bigr),\;\;
  f_{\mathrm{mid}}\bigl(\cdots,\, \mathbf{e}\bigr),\;\;
  f_{\mathrm{down2}}\bigl(\cdots,\, \mathbf{e}\bigr),\;\;
  f_{\mathrm{down1}}\bigl(\cdots,\, \mathbf{e}\bigr)
\Bigr)
$$

전체 noise 예측:

$$
\hat{\epsilon} = \epsilon_\theta(x_t,\, t,\, y)
$$

---

## 6. Loss에서의 역할

학습 objective:

$$
\mathcal{L}
= \mathbb{E}_{x_0,\, t,\, y,\, \epsilon}
\left[
\bigl\|\epsilon - \epsilon_\theta(x_t,\, t,\, y)\bigr\|_2^2
\right]
$$

condition $y$ 가 고정될 때 모델은 **해당 label에 대응하는 $x_0$ 분포의 noise 방향**을 학습한다.

$$
\epsilon_\theta(x_t,\, t,\, y)
\;\approx\;
-\sqrt{1 - \bar{\alpha}_t}\,
\nabla_{x_t} \log q(x_t \mid x_0^{(y)})
$$

---

## 7. Condition이 분포를 분리하는 원리

condition $y$ 는 **score function의 조건부 분포를 label별로 분리**하는 역할을 한다.

$$
p_\theta(x_{t-1} \mid x_t,\, y)
\;\neq\;
p_\theta(x_{t-1} \mid x_t,\, y')
\qquad (y \neq y')
$$

즉 reverse process가 label마다 **다른 방향으로 denoising**하도록 유도한다.

$$
p_\theta(x_0 \mid y) = \int p_\theta(x_0 \mid x_T,\, y)\, p(x_T)\, dx_T
$$

condition이 없는 unconditional 모델은 모든 label에 대한 **주변 분포(marginal)** 를 학습한다.

$$
p_\theta(x_0) = \sum_{y} p(y)\, p_\theta(x_0 \mid y)
$$

---

## 8. 조건 변화에 따른 동작

| 입력 condition | 모델 동작 |
|----------------|-----------|
| 정확한 label $y$ | $y$ 에 해당하는 hash 이미지 패턴으로 denoising |
| 잘못된 label $y'$ | 다른 hash label의 이미지 방향으로 denoising |
| null label $\varnothing$ (CFG) | "label 없음" 방향 — 평균적인 이미지 생성 |
| label 없음 (unconditional) | 전체 데이터 분포의 평균 방향으로 denoising |

---

## 9. 흐름 요약

```
label y
  │
  ▼
nn.Embedding(y)  →  e_c(y) ∈ R^d
                          │
                          ▼
  t  →  SinusoidalEmb → MLP  →  e_t(t) ∈ R^d
                          │
                          ▼
               e = e_t(t) + e_c(y)
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    down1(x, e)      mid(x, e)        up1(x, e)   ...
         │                                 │
         └────────── skip connection ───────┘
                          │
                          ▼
               output conv  →  ε̂  (predicted noise)
```
