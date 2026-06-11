# 프로젝트의 개요
## 프로젝트의 목표

**Diffusion Model을 사용해서 주어진 Hash 값의 원본 Message를 복구할 수 있는가**에 대한 개념 검증  

## 프로젝트의 배경 지식
### Diffusion Model
Diffusion Model은 원본 데이터에 노이즈를 점진적으로 추가하는 과정과, 노이즈가 섞인 데이터를 다시 원본에 가까운 형태로 되돌리는 과정을 학습하는 생성 모델이다.  
학습이 끝난 뒤에는 순수한 가우시안 노이즈에서 시작해 여러 단계의 denoising을 반복하면서 새로운 데이터를 생성한다.

**Forward Process**  
Forward Process는 원본 데이터 $x_0$에 timestep $t$마다 가우시안 노이즈를 추가하는 과정이다.  
각 단계에서 추가되는 노이즈의 크기는 $\beta_t$로 조절되며, 전체 노이즈 증가 방식은 noise schedule에 의해 결정된다.

$$
\alpha_t = 1 - \beta_t
$$

$$
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

$t$가 커질수록 $x_t$는 원본 정보보다 노이즈를 더 많이 포함한다.  
충분히 큰 timestep $T$에서는 $x_T$가 거의 순수한 가우시안 노이즈와 비슷한 분포가 된다.

**Backward Process**  
Backward Process는 노이즈가 섞인 데이터 $x_t$에서 이전 단계의 데이터 $x_{t-1}$를 추정하는 과정이다.  
일반적으로 모델은 $x_t$에 포함된 노이즈 $\epsilon$을 예측하도록 학습된다.

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$

여기서 $\epsilon_\theta$는 학습 가능한 denoising network를 의미한다.  
이미지 생성에서는 U-Net 계열 구조가 자주 사용되며, timestep 정보를 반영하기 위해 sinusoidal embedding이나 learned embedding을 함께 입력한다.

**Training**  
훈련은 다음 절차로 진행된다.

1. 원본 데이터 $x_0$를 샘플링한다.
2. timestep $t$를 무작위로 선택한다.
3. 가우시안 노이즈 $\epsilon$을 샘플링한다.
4. Forward Process를 통해 노이즈가 섞인 데이터 $x_t$를 만든다.
5. 모델이 $x_t$와 $t$를 입력받아 $\epsilon$을 예측하도록 학습한다.

대표적인 손실 함수는 실제 노이즈와 예측 노이즈 사이의 평균제곱오차(MSE)이다.

$$
L = \mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

이 학습 방식은 모델이 원본 데이터를 직접 맞히는 대신, 각 노이즈 단계에서 제거해야 할 노이즈의 방향을 학습하게 만든다.

**Sampling**  
Sampling은 학습된 denoising network를 사용해 노이즈를 데이터로 변환하는 과정이다.

1. $x_T \sim \mathcal{N}(0, I)$에서 시작한다.
2. timestep을 $T$부터 $1$까지 역순으로 진행한다.
3. 모델이 현재 단계의 노이즈를 예측한다.
4. 예측된 노이즈를 제거해 $x_{t-1}$를 계산한다.
5. 마지막 단계에서 생성 결과 $x_0$를 얻는다.

DDPM은 각 reverse step에서 확률적 샘플링을 수행하므로 다양한 결과를 생성할 수 있다.  
DDIM은 더 적은 step으로 빠르게 샘플링하거나 결정론적인 생성 경로를 사용할 수 있는 방법이다.

**Conditional Diffusion**  
Conditional Diffusion은 생성 과정에 조건 정보 $c$를 함께 입력해 원하는 방향으로 결과를 제어하는 방식이다.  
조건 정보에는 class label, text embedding, image embedding, segmentation map 등이 사용될 수 있다.

$$
\epsilon_\theta(x_t, t, c) \approx \epsilon
$$

조건부 Diffusion Model은 조건 $c$를 반영해 노이즈 제거 방향을 조정한다.  
Text-to-image 모델처럼 텍스트 설명을 기반으로 이미지를 생성하는 시스템이 대표적인 예이다.

### Hash Algorithm

Hash Algorithm은 임의 길이의 입력 데이터를 고정 길이의 출력값으로 변환하는 함수이다.  
이 출력값은 hash value, digest, fingerprint 등으로 불리며, 원본 데이터의 요약값처럼 사용된다.

$$
H: \{0, 1\}^{*} \rightarrow \{0, 1\}^{n}
$$

여기서 입력은 임의 길이의 bit string이고, 출력은 $n$ bit의 고정 길이 digest이다.  
예를 들어 SHA-256은 입력 길이와 관계없이 항상 256-bit hash value를 생성한다.

**기본 동작 과정**  
Hash Algorithm은 일반적으로 다음 과정을 거친다.

1. 입력 데이터를 byte 또는 bit 단위로 해석한다.
2. 알고리즘의 block size에 맞도록 padding을 추가한다.
3. 입력을 고정 크기의 block으로 나눈다.
4. 각 block을 compression function에 순차적으로 입력한다.
5. 마지막 internal state를 digest로 출력한다.

대부분의 암호학적 Hash Algorithm은 내부 상태를 반복적으로 갱신하는 구조를 가진다.  
각 block은 이전 block까지의 internal state와 함께 처리되며, 최종 digest는 전체 입력의 영향을 반영한다.

**일방향성**  
암호학적 Hash Algorithm은 대표적인 일방향 함수로 설계된다.  
입력 $x$로부터 $H(x)$를 계산하는 것은 빠르지만, 주어진 $H(x)$만 보고 원래 입력 $x$를 찾는 것은 계산적으로 매우 어렵다.

이 성질은 password 저장, 파일 무결성 검증, digital signature, blockchain 등에서 중요하게 사용된다.  
단, Hash Algorithm은 암호화와 다르다. 암호화는 key를 사용해 복호화가 가능하지만, Hash Algorithm은 원본 복원을 목적으로 설계되지 않는다.

**Avalanche Effect**  
Avalanche effect는 입력의 아주 작은 변화가 출력 전체에 큰 변화를 만드는 성질이다.  
예를 들어 입력에서 1 bit만 바뀌어도 digest의 많은 bit가 달라져야 한다.

이 성질이 강할수록 출력 digest만 보고 입력 간의 유사성을 추정하기 어렵다.  
따라서 좋은 Hash Algorithm은 비슷한 입력이라도 전혀 다른 출력처럼 보이도록 설계된다.

**충돌 저항성**  
Hash Algorithm은 임의 길이의 입력을 고정 길이의 출력으로 압축하므로, 이론적으로 서로 다른 두 입력이 같은 digest를 갖는 충돌은 반드시 존재한다.  
하지만 안전한 Hash Algorithm은 실제로 충돌을 찾는 것이 계산적으로 매우 어렵도록 설계된다.

주요 보안 성질은 다음과 같다.

- Preimage resistance: digest $y$가 주어졌을 때 $H(x)=y$를 만족하는 입력 $x$를 찾기 어려워야 한다.
- Second preimage resistance: 특정 입력 $x$가 주어졌을 때 $H(x)=H(x')$인 다른 입력 $x'$를 찾기 어려워야 한다.
- Collision resistance: 서로 다른 두 입력 $x, x'$에 대해 $H(x)=H(x')$를 만족하는 쌍을 찾기 어려워야 한다.

**대표 알고리즘**  
MD5, SHA-1, SHA-2, SHA-3 등이 대표적인 Hash Algorithm이다.  
MD5와 SHA-1은 현재 충돌 공격이 알려져 있어 보안 목적에는 권장되지 않는다.  
SHA-256과 SHA-512는 SHA-2 계열에 속하며, 현재도 널리 사용되는 암호학적 Hash Algorithm이다.

## 프로젝트의 기반 아이디어
**Hash Algorithm의 내부 상태를 갱신하는 구조와 Diffusion Model에서 Forward Process의 유사성**에 입각  
Hash Algorithm에서 내부 상태를 1 차례 갱신 $\sim$ Diffusion Model의 Forward Process에서 1 timestep  
$\therefore$ Hash Algorithm의 연산 결과인 Hash 값을 Diffusion Model을 이용해서 원본 Message롤 복원할 수 있는가

