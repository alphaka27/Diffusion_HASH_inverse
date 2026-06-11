# 실험 워크플로우

## 실험 전제
- 원본 메시지의 길이를 알고 있다.  
- Salt와 Pepper와 같은 해시 입력 보강값이 적용되어 있지 않다.

## 1. 데이터 생성

### 1.1 메시지 생성
Diffusion Model 학습에 사용할 원본 메시지를 생성한다.  
메시지는 bit 단위 또는 character 단위로 생성할 수 있으며, character mode에서는 사용할 문자 후보군을 별도로 지정한다.

**입력**

- 원본 메시지의 bit 길이
- 생성 mode: bit 또는 character
- character 후보군: character mode에서 사용
- 생성할 sample 수

**출력**

- 원본 메시지
- 메시지 길이 metadata

### 1.2 Hash 연산 및 로그 수집
생성된 원본 메시지에 대해 Hash Algorithm을 수행한다.  
최종 hash value뿐 아니라, Diffusion Model 학습에 활용할 수 있는 중간 연산 로그도 함께 저장한다.

**입력**

- 원본 메시지
- Hash Algorithm 종류
- Hash 설정값: word size, block size, byte order 등

**출력**

- 최종 hash value
- Hash 연산 중간 과정 로그
- 메시지와 hash value를 연결하는 metadata

중간 과정 로그는 JSON 형식으로 저장한다.  
이 로그는 이후 이미지 또는 행렬 형태로 인코딩되어 Diffusion Model의 학습 데이터로 사용될 수 있다.

## 2. 이미지 인코딩

### 2.1 RGB 인코딩
Hash 연산 로그 또는 메시지 byte sequence 중 학습에 사용할 부분을 RGB 이미지로 변환한다.  
구체적인 RGB encoding/decoding 규칙은 [Encoding_kr.md](./Encoding_kr.md)에 정의한다.

**Encoding**

```text
Byte sequence -> RGB value sequence -> PNG image
```

**Decoding**

```text
PNG image -> RGB value sequence -> Byte sequence
```

RGB 인코딩은 byte 값을 RGB 공간의 특정 영역에 대응시키는 방식이다.  
Diffusion Model이 생성한 이미지에서 RGB 값을 추출한 뒤, 동일한 규칙을 사용해 원래의 byte sequence로 복원한다.

### 2.2 Matrix 인코딩
Byte 또는 숫자 단위의 데이터를 행렬 구조의 이미지로 변환한다.  
Matrix 인코딩은 RGB 값 자체보다 공간적 패턴을 학습 대상으로 삼는 방식이다.

**Encoding**

```text
Byte sequence -> Matrix representation -> PNG image
```

**Decoding**

```text
PNG image -> Matrix representation -> Byte sequence
```

Matrix 구조는 [Encoding_kr.md](./Encoding_kr.md)에 정의된 규칙에 따라 생성한다.  
추론 시에는 생성된 이미지에서 행렬 구조를 복원하고, 복원된 패턴을 다시 byte 값으로 디코딩한다.

## 3. 학습 데이터셋 구성
Diffusion Model의 학습 sample은 이미지와 조건 정보(condition)의 쌍으로 구성한다.

**이미지 데이터**

- 원본 메시지 또는 Hash 연산 로그를 인코딩한 PNG 파일
- RGB 인코딩 이미지 또는 Matrix 인코딩 이미지

**조건 정보**

- 원본 메시지에 대한 hash value
- 원본 메시지의 bit 길이
- Hash Algorithm 종류
- 필요한 경우 Hash 연산 단계 또는 로그 위치 정보

데이터셋은 이미지 파일과 metadata를 함께 관리해야 한다.  
각 이미지가 어떤 원본 메시지, hash value, bit 길이와 대응되는지 추적 가능해야 한다.

## 4. Text-Based Diffusion Model 워크플로우

### 4.1 개요
Text-Based Diffusion Model은 text-to-image generation 구조를 차용한다.  
모델은 hash value와 메시지 길이를 조건으로 입력받고, 해당 조건에 대응되는 인코딩 이미지를 생성하도록 학습한다.

학습 대상 이미지는 원본 메시지 또는 Hash 연산 로그를 인코딩한 PNG 파일이다.  
조건 정보는 해당 이미지와 연결된 hash value 및 메시지 길이이다.

### 4.2 학습 파이프라인
학습 단계에서는 인코딩 이미지에 Forward Process를 적용하고, 조건 정보를 사용해 Reverse Process를 학습한다.

1. 원본 메시지 또는 Hash 연산 로그를 PNG 이미지로 인코딩한다.
2. PNG 이미지에 timestep별 가우시안 노이즈를 추가한다.
3. hash value와 메시지 길이를 condition으로 입력한다.
4. U-Net 또는 denoising network가 추가된 노이즈를 예측하도록 학습한다.
5. 실제 노이즈와 예측 노이즈 사이의 손실을 최소화한다.

이 과정에서 모델은 주어진 조건이 어떤 이미지 분포와 연결되는지를 학습한다.  
학습이 끝나면 동일한 조건을 입력했을 때 해당 조건에 맞는 인코딩 이미지를 생성할 수 있다.

### 4.3 추론 파이프라인
추론 단계에서는 hash value와 메시지 길이를 조건으로 사용해 이미지를 생성한다.

1. 입력 hash value를 condition embedding으로 변환한다.
2. 원본 메시지의 bit 길이를 함께 condition으로 입력한다.
3. 순수한 가우시안 노이즈에서 reverse sampling을 시작한다.
4. 학습된 denoising network를 통해 인코딩 이미지를 생성한다.
5. 생성된 이미지를 RGB 또는 Matrix decoding 규칙에 따라 byte sequence로 복원한다.
6. 복원된 byte sequence를 후보 메시지로 사용한다.

생성 결과는 확률적 샘플링의 영향을 받으므로, 같은 condition에서도 여러 후보가 생성될 수 있다.  
따라서 추론 결과는 단일 출력이 아니라 후보군으로 관리하는 것이 적합하다.

## 5. Process-Based Diffusion Model 워크플로우

### 5.1 개요
Process-Based Diffusion Model은 최종 hash value만 사용하는 대신, Hash Algorithm의 중간 연산 과정을 학습 대상으로 삼는 접근이다.  
Hash 연산에서 내부 상태가 단계적으로 갱신되는 구조를 Diffusion Model의 timestep 흐름과 대응시켜 분석한다.

### 5.2 학습 방향
Hash Algorithm의 중간 상태, round별 출력, block별 업데이트 로그를 sequence 또는 image representation으로 변환한다.  
모델은 이 representation을 바탕으로 연산 과정의 구조적 패턴을 학습한다.

**학습 데이터 후보**

- round별 internal state
- block별 compression 결과
- word schedule 또는 message schedule
- 최종 hash value와 중간 상태의 대응 관계

### 5.3 추론 방향
추론 단계에서는 최종 hash value 또는 일부 중간 상태를 조건으로 입력해, 가능한 연산 과정 또는 원본 표현 후보를 생성한다.  
생성된 후보는 다시 decoding 과정을 거쳐 byte sequence로 복원한다.

Process-Based 방식은 Text-Based 방식보다 학습 데이터 구성이 복잡하지만, Hash Algorithm의 내부 구조를 더 많이 활용할 수 있다는 장점이 있다.

## 6. 결과 검증
Diffusion Model이 생성한 결과는 최종적으로 Hash Algorithm을 다시 적용해 검증한다.

1. 생성 이미지를 byte sequence로 디코딩한다.
2. 디코딩된 byte sequence를 후보 메시지로 해석한다.
3. 후보 메시지에 동일한 Hash Algorithm을 적용한다.
4. 생성 조건으로 사용한 hash value와 계산 결과를 비교한다.
5. 일치하는 경우 성공 후보로 기록한다.

검증 결과는 후보 메시지, 생성 이미지, condition, hash value, decoding 성공 여부와 함께 저장한다.  
이 정보는 모델 성능 평가와 후속 실험 분석에 사용된다.
