# CLI 실행 방법

## 기본 실행 형태

패키지를 설치하지 않은 개발 환경에서는 모듈 기준으로 실행한다.

```bash
python -m diffusion_hash_inv.hash_main --help
python -m diffusion_hash_inv.hash_main -i 10 -l 128 --hash-alg md5
```

editable 설치 후에는 `pyproject.toml`에 등록된 콘솔 스크립트를 사용할 수 있다.

```bash
python -m pip install -e .
diffhash --help
diffhash hash --help
diffhash -i 10 -l 128 --hash-alg md5
```

## 중앙 CLI 서브커맨드

| 명령 | 설명 |
| --- | --- |
| `diffhash hash` | 해시 생성/검증 실행 |
| `diffhash mlx-toy` | synthetic prototype 기반 MLX DDPM demo 실행 |
| `diffhash mlx-conditional` | 생성 이미지와 JSON 로그를 사용하는 MLX conditional DDPM 학습 |

기존 호환성을 위해 `diffhash -i 10 -l 128`처럼 서브커맨드 없이 해시 옵션을 바로 넘겨도 `diffhash hash -i 10 -l 128`과 동일하게 동작한다.

## 메인 해시 생성 CLI

```bash
python -m diffusion_hash_inv.hash_main [options]
diffhash hash [options]
```

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `--hash-alg`, `--hash_alg` | 사용할 해시 알고리즘 | `md5` |
| `-l`, `--length` | 입력 비트 길이 | `256` |
| `-e`, `--exponentiation` | 입력 비트 길이를 `2 ** N`으로 지정 | 없음 |
| `-i`, `--iteration` | 실행 반복 횟수 | `0` |
| `-b`, `--bit` | 비트 문자열 입력 모드 | 기본 |
| `-m`, `--message` | 메시지 입력 모드 | 현재 실행부 미구현 |
| `--random` | 반복마다 난수 입력 생성 | 기본 |
| `--sequential` | 반복 인덱스 값을 입력으로 사용 | 꺼짐 |
| `-v`, `--verbose` | 상세 로그 출력 | 꺼짐 |
| `-c`, `--clear` | 실행 전 `data`, `output` 정리 | 꺼짐 |
| `--make-image` | 해시 로그 생성 후 RGB 이미지 생성 | 꺼짐 |

`-l`과 `-e`는 동시에 사용할 수 없다. `--sequential`을 사용해도 `-i` 기본값은 `0`이므로 실제 생성하려면 반복 횟수를 명시한다.

### 예시

128비트 랜덤 입력 10개를 MD5로 처리한다.

```bash
python -m diffusion_hash_inv.hash_main -i 10 -l 128 --hash-alg md5
```

16비트 입력 공간의 앞 16개 값을 순차 처리한다.

```bash
python -m diffusion_hash_inv.hash_main -i 16 -l 16 --hash-alg md5 --sequential
```

입력 길이를 `2 ** 8 = 256`비트로 지정한다.

```bash
python -m diffusion_hash_inv.hash_main -i 5 -e 8 --hash-alg md5
```

기존 출력물을 정리하고 새로 실행한다.

```bash
python -m diffusion_hash_inv.hash_main -c -i 10 -l 128 --hash-alg md5
```

해시 로그 생성 후 이미지 생성까지 실행한다.

```bash
python -m diffusion_hash_inv.hash_main -i 10 -l 128 --hash-alg md5 --make-image
```

## 출력 위치

메인 CLI는 기본적으로 프로젝트 루트 기준 경로를 사용한다.

| 데이터 | 위치 |
| --- | --- |
| 생성 입력 바이너리 | `data/binary/` |
| 해시 JSON 로그 | `output/json/<program-start-time>/` |
| 이미지 출력 | `data/images/`, `output/images/` |

## 랜덤 비트 생성기

해시 실행 없이 랜덤 비트 파일만 생성한다.

```bash
python -m diffusion_hash_inv.generator.random_n_bits -l 512 -i 3
```

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `-l`, `--length` | 생성할 비트 길이 | `512` |
| `-e`, `--exponentiation` | 비트 길이를 `2 ** N`으로 지정 | 없음 |
| `-i`, `--iterations` | 생성 반복 횟수 | `1` |
| `-v`, `--verbose` | 상세 출력 | 켜짐 |
| `-q`, `--quiet` | 출력 억제 | 꺼짐 |
| `-c`, `--clear` | 실행 전 출력 정리 | 꺼짐 |
| `-C`, `--no-clear` | 출력 정리 안 함 | 기본 |

## 랜덤 문자 생성기

해시 실행 없이 랜덤 문자 파일만 생성한다. 길이 옵션은 비트 단위이며 내부적으로 `length / 8`개의 ASCII 문자를 만든다.

```bash
python -m diffusion_hash_inv.generator.random_n_char -l 512 -i 3
```

옵션은 랜덤 비트 생성기와 동일하다.

## 빠른 확인 명령

CLI import와 기본 실행 경로만 확인한다.

```bash
python -m diffusion_hash_inv.hash_main -i 0
```

전체 테스트를 실행한다.

```bash
python -m pytest -q
```

## MLX 모델 CLI

### Synthetic MLX toy DDPM

실제 데이터 없이 MLX 학습 루프와 sampling을 확인한다.

```bash
python -m diffusion_hash_inv.models.diffusion_with_mlx \
  --device cpu \
  --train-steps 50 \
  --timesteps 20
```

설치 후 중앙 CLI로 실행할 수 있다.

```bash
diffhash mlx-toy \
  --device cpu \
  --train-steps 50 \
  --timesteps 20
```

### 실제 데이터셋 기반 MLX conditional DDPM

`data/images/<run-id>/message.png`와 `output/json/**/<run-id>.json`을 매칭해 최종 해시 라벨 조건으로 학습한다.

```bash
python -m diffusion_hash_inv.models.conditional_diffusion_mlx \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_mlx \
  --train-steps 500 \
  --timesteps 100 \
  --batch-size 32 \
  --device cpu
```

설치 후 중앙 CLI로 실행할 수 있다.

```bash
diffhash mlx-conditional \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_mlx \
  --train-steps 500 \
  --timesteps 100 \
  --batch-size 32 \
  --device cpu
```
