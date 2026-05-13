# Diffusion Model 구현 현황

## 구현 목록

| 파일 | 종류 | 프레임워크 | 입력/조건 | 상태 |
| --- | --- | --- | --- | --- |
| `models/conditional_diffusion.py` | Conditional DDPM | PyTorch | `message.png`, 최종 해시 라벨 | 실제 데이터셋 학습 파이프라인 |
| `models/unconditional_ddpm.py` | Unconditional DDPM | PyTorch | `message.png` | 라벨 없는 생성 모델 |
| `models/guided_conditional_diffusion.py` | Guided Conditional DDPM | PyTorch | 최종 해시 라벨 | classifier guidance, classifier-free guidance |
| `models/loop_conditioned_diffusion.py` | Loop-conditioned DDPM | PyTorch | MD5 Step 4 loop-state tensor | 구조화된 루프 상태 조건 |
| `models/diffusion_with_mlx.py` | Conditional DDPM toy | MLX | synthetic class prototype | MLX 동작 확인용 self-contained 예제 |
| `models/conditional_diffusion_mlx.py` | Conditional DDPM | MLX | `message.png`, 최종 해시 라벨 | 새 MLX 실제 데이터셋 학습 파이프라인 |
| `models/ddim.py` | DDIM placeholder | 없음 | 없음 | placeholder |
| `models/ddpm.py` | DDPM placeholder | 없음 | 없음 | 빈 파일 |

## PyTorch 계열 요약

### Conditional DDPM

`conditional_diffusion.py`는 생성된 이미지와 JSON 로그를 매칭해 최종 해시 값을 조건 라벨로 사용한다. 기본 모델은 U-Net 계열 denoiser이고, DDPM noise scheduler를 사용한다.

지원하는 temporal conditioning:

| 모드 | 의미 |
| --- | --- |
| `class` | 라벨 embedding만 사용 |
| `loop-sinusoidal` | loop index sinusoidal encoding |
| `loop-structured` | loop index와 loop 구간 정보를 MLP로 결합 |
| `loop-sequence` | loop sequence 전체를 Transformer encoder로 조건화 |

### Unconditional DDPM

`unconditional_ddpm.py`는 JSON 라벨을 쓰지 않고 `data/images/<run-id>/message.png`만 학습한다. 데이터 분포 자체를 학습하는 baseline으로 사용한다.

### Guided Conditional DDPM

`guided_conditional_diffusion.py`는 조건부 생성 품질을 높이기 위한 guidance 변형이다.

| 모드 | 설명 |
| --- | --- |
| `classifier` | noisy image classifier의 gradient로 reverse sampling 보정 |
| `classifier-free` | label dropout으로 conditional/unconditional 예측을 함께 학습하고 sampling 때 섞음 |

### Loop-conditioned DDPM

`loop_conditioned_diffusion.py`는 JSON의 `Logs/4th Step/1st Round` 아래 loop state를 연속 tensor로 변환한다. diffusion timestep을 loop state sequence에 매핑해 denoiser에 제공한다.

## MLX 계열 요약

### 기존 MLX toy

`diffusion_with_mlx.py`는 실제 프로젝트 데이터가 없어도 실행할 수 있는 synthetic prototype 기반 conditional DDPM이다. MLX 설치, optimizer, scheduler, sampling 경로 확인용이다.

실행:

```bash
python -m diffusion_hash_inv.models.diffusion_with_mlx \
  --device cpu \
  --train-steps 50 \
  --timesteps 20
```

또는 설치된 CLI에서:

```bash
diffhash mlx-toy --device cpu --train-steps 50 --timesteps 20
```

### 새 MLX Conditional DDPM

`conditional_diffusion_mlx.py`는 실제 생성 이미지와 JSON 로그를 사용하는 MLX 구현이다.

데이터 구조:

```text
data/images/<run-id>/message.png
output/json/**/<run-id>.json
```

라벨:

JSON의 `Generated hash`, `Correct   hash`, `Correct hash` 중 먼저 발견되는 값을 최종 해시 라벨로 사용한다.

실행:

```bash
python -m diffusion_hash_inv.models.conditional_diffusion_mlx \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_mlx \
  --image-size 32 \
  --channels 1 \
  --fit-mode pad \
  --train-steps 500 \
  --timesteps 100 \
  --batch-size 32 \
  --device cpu
```

또는 설치된 CLI에서:

```bash
diffhash mlx-conditional \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_mlx \
  --image-size 32 \
  --channels 1 \
  --fit-mode pad \
  --train-steps 500 \
  --timesteps 100 \
  --batch-size 32 \
  --device cpu
```

빠른 smoke run:

```bash
python -m diffusion_hash_inv.models.conditional_diffusion_mlx \
  --data-root data/images \
  --json-root output/json \
  --train-steps 1 \
  --timesteps 4 \
  --batch-size 2 \
  --sample-count 2 \
  --image-size 8 \
  --hidden-dim 16 \
  --time-dim 8
```

출력:

| 파일 | 설명 |
| --- | --- |
| `output/conditional_diffusion_mlx/config.json` | 학습 설정 |
| `output/conditional_diffusion_mlx/label_map.json` | 최종 해시 라벨과 class index 매핑 |
| `output/conditional_diffusion_mlx/samples.png` | 생성 샘플 grid |

## 동작 수식 정리

아래 수식은 현재 코드가 구현한 DDPM 계열 모델의 공통 동작을 기준으로 한다. PyTorch 구현의 timestep index는 `t = 0, ..., T-1`이다.

### 공통 noise schedule

`DDPMNoiseScheduler`는 beta schedule에서 다음 계수를 만든다.

$$
\beta_t \in (0, 1), \qquad
\alpha_t = 1 - \beta_t, \qquad
\bar{\alpha}_t = \prod_{s=0}^{t} \alpha_s
$$

posterior variance는 코드에서 다음처럼 계산한다.

$$
\tilde{\beta}_t =
\beta_t \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t},
\qquad
\bar{\alpha}_{-1} = 1
$$

### Forward process

학습 이미지 `x_0`에 Gaussian noise `epsilon`을 섞어 `x_t`를 만든다.

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
q(x_t \mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$

구현에서 직접 쓰는 형태는 다음과 같다.

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

### Noise prediction loss

모델은 `x_t`에서 제거할 noise를 예측한다.

$$
\hat{\epsilon}
=
\epsilon_\theta(x_t, t, c)
$$

학습 loss는 noise prediction MSE다.

$$
\mathcal{L}_{DDPM}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon - \epsilon_\theta(x_t,t,c)
\right\|_2^2
\right]
$$

모델별 condition `c`는 다음과 같다.

| 모델 | condition `c` |
| --- | --- |
| Unconditional DDPM | 없음 |
| Conditional DDPM, `class` mode | final hash label `y` |
| Conditional DDPM, temporal modes | loop metadata `m = [loop_idx, loop_count, loop_start, loop_end]` |
| Guided Conditional DDPM | final hash label `y`, 또는 null label |
| Loop-conditioned DDPM | MD5 loop-state tensor `S` |
| MLX Conditional DDPM | final hash label `y` |

### Reverse process

sampling은 pure noise에서 시작한다.

$$
x_T \sim \mathcal{N}(0, I)
$$

각 reverse step에서 모델의 noise 예측값으로 평균을 계산한다.

$$
\mu_\theta(x_t,t,c)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t,c)
\right)
$$

`t > 0`이면 posterior variance를 사용해 다음 sample을 만든다.

$$
x_{t-1}
=
\mu_\theta(x_t,t,c)
+
\sqrt{\tilde{\beta}_t}z,
\qquad
z \sim \mathcal{N}(0, I)
$$

`t = 0`에서는 noise를 더하지 않고 평균만 반환한다.

$$
x_0 = \mu_\theta(x_0,t=0,c)
$$

### Conditional DDPM condition embedding

기본 conditional U-Net은 timestep embedding과 condition embedding을 더해 각 residual block에 넣는다.

$$
h(t,c) = e_t(t) + e_c(c)
$$

`class` mode에서는 final hash label `y`를 embedding한다.

$$
e_c(c) = \mathrm{Embedding}(y)
$$

temporal conditioning mode에서는 label 대신 loop metadata를 쓴다.

`loop-sinusoidal`:

$$
e_c(c) = \mathrm{PE}(loop\_idx)
$$

`loop-structured`:

$$
e_c(c)
=
\mathrm{PE}(loop\_idx)
+
\mathrm{MLP}_{start}(loop\_start)
+
\mathrm{MLP}_{end}(loop\_end)
$$

`loop-sequence`:

$$
E = \mathrm{TransformerEncoder}(\mathrm{LoopTokens})
$$

$$
e_c(c) = \mathrm{MLP}(E_{loop\_idx})
$$

### Classifier-free guidance

classifier-free guidance 학습에서는 확률 `p_drop`로 label을 null label로 바꾼다.

$$
c' =
\begin{cases}
c, & \text{with probability } 1-p_{drop} \\
\varnothing, & \text{with probability } p_{drop}
\end{cases}
$$

sampling에서는 conditional prediction과 unconditional prediction을 섞는다.

$$
\epsilon_{uncond}
=
\epsilon_\theta(x_t,t,\varnothing)
$$

$$
\epsilon_{cond}
=
\epsilon_\theta(x_t,t,c)
$$

$$
\epsilon_{cfg}
=
\epsilon_{uncond}
+
w(\epsilon_{cond}-\epsilon_{uncond})
$$

이후 reverse mean은 `epsilon_cfg`로 계산한다.

$$
\mu_{cfg}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_{cfg}
\right)
$$

### Classifier guidance

classifier guidance에서는 noisy image classifier가 timestep별 label 확률을 학습한다.

$$
p_\phi(y \mid x_t,t)
=
\mathrm{softmax}(f_\phi(x_t,t))_y
$$

classifier loss는 cross entropy다.

$$
\mathcal{L}_{cls}
=
-\log p_\phi(y \mid x_t,t)
$$

sampling에서는 classifier gradient를 reverse mean에 더한다. 현재 구현은 denoiser에 고정 label `0`을 넣어 base reverse mean을 만든다.

$$
g =
\nabla_{x_t}
\log p_\phi(y \mid x_t,t)
$$

$$
\mu_{guided}
=
\mu_\theta(x_t,t,0)
+
\tilde{\beta}_t w g
$$

그 다음 일반 reverse sampling과 같이 진행한다.

$$
x_{t-1}
=
\mu_{guided}
+
\sqrt{\tilde{\beta}_t}z
$$

### Loop-conditioned DDPM

loop-conditioned 모델은 final hash label 대신 MD5 Step 4 loop state sequence를 condition으로 쓴다.

기본 condition tensor는 다음 형태다.

$$
S \in [-1,1]^{(L+2) \times W}
$$

기본값은 `L = 64`, `W = 4`라서 다음 shape가 된다.

$$
S \in [-1,1]^{66 \times 4}
$$

각 row는 다음 상태 중 하나다.

$$
S_k = [A_k, B_k, C_k, D_k]
$$

uint32 word는 다음 방식으로 정규화된다.

$$
\mathrm{norm}(v)
=
\frac{v}{2^{32}-1} \cdot 2 - 1
$$

diffusion timestep `t`는 loop state index로 매핑된다.

$$
k(t)
=
\left\lfloor
\frac{t \cdot (L+2)}{T}
\right\rfloor
$$

선택된 loop state는 timestep embedding, state value embedding, state position embedding으로 합쳐진다.

$$
h(t,S)
=
e_t(t)
+
e_s(S_{k(t)})
+
e_p(k(t))
$$

모델은 다음 noise를 예측한다.

$$
\hat{\epsilon}
=
\epsilon_\theta(x_t,t,S)
$$

학습 loss는 공통 DDPM MSE와 같다.

$$
\mathcal{L}_{loop}
=
\mathbb{E}
\left[
\left\|
\epsilon - \epsilon_\theta(x_t,t,S)
\right\|_2^2
\right]
$$

### MLX Conditional DDPM

MLX 구현은 이미지를 flatten vector로 다룬다.

$$
x_0 \in \mathbb{R}^{D},
\qquad
D = C \cdot H \cdot W
$$

MLP denoiser 입력은 noised vector, timestep embedding, final hash label embedding을 concatenate한 것이다.

$$
h =
\mathrm{concat}
\left(
x_t,
e_t(t),
e_y(y)
\right)
$$

$$
\hat{\epsilon}
=
\mathrm{MLP}_\theta(h)
$$

loss와 reverse sampling 수식은 PyTorch conditional DDPM과 동일하다.

## 선택 기준

| 목적 | 권장 구현 |
| --- | --- |
| 실제 이미지와 최종 해시 조건 학습 | `conditional_diffusion.py` 또는 `conditional_diffusion_mlx.py` |
| 라벨 없는 baseline | `unconditional_ddpm.py` |
| 조건 강도를 조절한 sampling | `guided_conditional_diffusion.py` |
| MD5 내부 loop state를 조건으로 사용 | `loop_conditioned_diffusion.py` |
| MLX 설치/기본 학습 루프 확인 | `diffusion_with_mlx.py` |
