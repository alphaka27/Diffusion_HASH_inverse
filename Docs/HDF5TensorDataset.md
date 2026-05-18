# HDF5 Tensor Dataset

이 문서는 JSON hash log에서 생성한 이미지를 HDF5 tensor shard로 저장하고, PyTorch `Dataset` / `DataLoader`로 읽는 방법을 정리한다.

## 목적

기존 PNG 출력은 log 1개당 `message.png`와 step 이미지 등 여러 파일을 만든다. 대량 실행에서는 파일 수가 빠르게 늘어나므로, 학습용 tensor를 HDF5 shard로 묶어 저장하면 다음 이점이 있다.

- 이미지 파일 수를 줄여 metadata I/O 부담을 낮춘다.
- PIL decode 과정을 학습 시점이 아니라 dataset 생성 시점으로 옮긴다.
- `C,H,W` tensor 형태를 바로 읽을 수 있다.
- shard 단위로 process 병렬 생성을 수행할 수 있다.

주의할 점은 HDF5가 단일 파일에 대한 다중 process 동시 쓰기에 적합하지 않다는 점이다. 이 구현은 worker마다 독립 shard 파일을 쓰고, 마지막에 `manifest.json`으로 shard 목록을 묶는다.

## 의존성

HDF5 생성과 로딩에는 `h5py`가 필요하다.

```bash
pip install -e ".[hdf5]"
```

PyTorch `DataLoader`까지 사용하려면 train extra도 함께 설치한다.

```bash
pip install -e ".[train,hdf5]"
```

## 생성 API

`HDF5Maker`는 JSON log를 읽고, 각 log에서 생성 가능한 이미지를 tensor로 변환해 shard에 저장한다.

```python
from diffusion_hash_inv.config import Byte2RGBConfig
from diffusion_hash_inv.logger import LogStream
from diffusion_hash_inv.utils import FileIO, HDF5Maker

# runtime_config는 Docs/Workflow.md의 RuntimeConfig 예시처럼 구성한다.
io_controller = FileIO(runtime_config.main, runtime_config.output)
rgb_config = Byte2RGBConfig(seed_flag=False, input_seed=42)

log_paths = list(LogStream(runtime_config.output.output_dir / "json").iter_files())

hdf5_maker = HDF5Maker(runtime_config, io_controller)
hdf5_maker.main(
    logs=log_paths,
    rgb_config=rgb_config,
    workers=4,
    output_name="hash_tensors",
    shard_size=1000,
    include_paths=None,
    preserve_log_hierarchy=True,
)
```

주요 인자:

| 인자 | 설명 |
| --- | --- |
| `logs` | 변환할 JSON log 경로 목록. `None`이면 현재 설정에 맞는 최신 log를 찾는다. |
| `rgb_config` | log 값을 RGB 이미지로 바꾸는 설정. |
| `workers` | shard 생성 process 수. `1`이면 순차 실행한다. |
| `output_name` | `data/tensor_datasets/<output_name>/` 아래에 dataset을 만든다. |
| `shard_size` | shard 1개에 넣을 JSON log 수. |
| `include_paths` | 저장할 이미지 path 필터. `None`이면 message와 모든 log tensor를 저장한다. |
| `channels` | 출력 channel 수. 기본값은 `3`이다. |
| `normalize` | `False`이면 `uint8 [0, 255]`, `True`이면 `float32 [-1, 1]`로 저장한다. `make_image_flag=True` 통합 경로는 `False`를 사용한다. |
| `compression` | HDF5 dataset 압축 방식. 기본값은 `gzip`이다. |
| `preserve_log_hierarchy` | `True`이면 HDF5 내부에 `logs/<source_log>/Message/Hex`, `logs/<source_log>/Logs/...` 계층을 만든다. |

`include_paths=None`이면 log에서 만들 수 있는 모든 이미지 tensor를 저장한다. 그러나 step 이미지와 message 이미지는 shape가 다를 수 있으므로, 학습 batch를 바로 만들 목적이면 동일 shape만 선택하는 것이 안전하다.

## 출력 구조

예를 들어 `output_name="hash_tensors"`로 생성하면 다음 구조가 만들어진다.

```text
data/tensor_datasets/hash_tensors/
  manifest.json
  hash_tensors_000000.h5
  hash_tensors_000001.h5
  ...
```

각 `.h5` shard의 구조는 다음과 같다.

```text
records/
  000000/
    tensor
  000001/
    tensor
  ...
logs/
  <source-log-stem>/
    Message/
      Hex/
        tensor
    Logs/
      <Step>/
        <Round>/
          <Loop>/
            tensor
paths
source_logs
record_keys
```

- `records/<record_key>/tensor`: `C,H,W` tensor.
- `paths`: 원래 이미지 path. 예: `message.png`.
- `source_logs`: tensor가 만들어진 JSON log 파일명.
- `record_keys`: `records` 아래의 key 목록.
- `logs/<source-log-stem>/Message/Hex/tensor`: message를 encoding한 tensor.
- `logs/<source-log-stem>/Logs/.../tensor`: 기존 JSON log의 step/round/loop 계층을 따라 배치한 encoded log tensor.

`manifest.json`은 shard 파일 목록, 전체 tensor 개수, 생성 옵션, path 종류를 기록한다. Loader는 이 manifest를 기준으로 shard를 찾는다.

`logs/.../tensor`는 `records/<record_key>/tensor`에 대한 HDF5 hard-link다. 따라서 hierarchy view를 추가해도 tensor data가 중복 저장되지는 않는다.

## Loader API

HDF5 shard는 `HDF5TensorDataset` 또는 `create_hdf5_tensor_dataloader`로 읽는다.

```python
from diffusion_hash_inv.utils import HDF5TensorDataset, create_hdf5_tensor_dataloader

dataset = HDF5TensorDataset(
    "data/tensor_datasets/hash_tensors",
    include_paths=("message.png",),
    require_same_shape=True,
)

tensor, index = dataset[0]
metadata = dataset.metadata(0)

loader = create_hdf5_tensor_dataloader(
    "data/tensor_datasets/hash_tensors",
    batch_size=64,
    shuffle=True,
    num_workers=4,
    include_paths=("message.png",),
)

for tensors, indices in loader:
    # tensors shape: [B, C, H, W]
    pass
```

기본 반환값은 `(tensor, index_tensor)`이다. metadata까지 같이 받고 싶으면 `return_metadata=True`로 `HDF5TensorDataset`을 생성한다.

```python
dataset = HDF5TensorDataset(
    "data/tensor_datasets/hash_tensors",
    include_paths=("message.png",),
    return_metadata=True,
)

tensor, index, metadata = dataset[0]
print(metadata["source_log"], metadata["path"])
```

## Shape 제약

PyTorch 기본 collate는 batch 안의 tensor shape가 모두 같아야 한다. 그래서 학습용 loader는 다음처럼 동일 image type만 고르는 방식을 권장한다.

```python
include_paths=("message.png",)
```

`include_paths=None`은 모든 tensor를 inspection할 때는 유용하지만, 서로 다른 shape가 섞이면 `DataLoader` batch 생성에서 실패할 수 있다. 이 경우 선택지는 세 가지다.

- `include_paths`로 한 종류만 선택한다.
- 생성 단계에서 같은 크기로 resize된 tensor만 저장한다.
- custom `collate_fn`을 만들어 variable shape를 직접 처리한다.

## Multiprocessing 동작

`HDF5TensorDataset`은 pickling될 때 열린 HDF5 file handle을 유지하지 않는다. `DataLoader(num_workers>0)`를 사용할 때 각 worker process가 필요한 shard를 lazy-open한다. 따라서 loader는 다음 형태로 사용할 수 있다.

```python
loader = create_hdf5_tensor_dataloader(
    "data/tensor_datasets/hash_tensors",
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    include_paths=("message.png",),
)
```

`persistent_workers=True`는 epoch 사이에 worker를 재사용하므로 반복 학습에서는 유리할 수 있다. 단, dataset 경로의 HDF5 파일을 학습 중에 다시 쓰면 안 된다.

## Analyze Notebook 사용

`src/diffusion_hash_inv/analyze/analyze.ipynb`에는 `make_image_flag=True` 기반 PNG/HDF5 생성과 loader 확인 cell이 포함되어 있다.

```python
HDF5_WORKERS = IMAGE_WORKERS
HDF5_SHARD_SIZE = 1000
HDF5_INCLUDE_PATHS = None
HDF5_LOADER_INCLUDE_PATHS = ("message.png",)
```

위 설정은 JSON log 생성 후 PNG image 생성과 HDF5 tensor shard 생성을 process 병렬로 수행한다. HDF5 생성은 message와 모든 log tensor를 저장하고, loader 확인 cell은 batch shape를 맞추기 위해 `message.png`만 선택해 `DataLoader`를 만든다.

## 현재 한계

- 현재 DDPM trainer는 기본적으로 `data/images/<run-id>/message.png` PNG 입력을 사용한다.
- HDF5 loader는 학습 루프에 연결할 수 있는 PyTorch `DataLoader`를 제공하지만, 기존 trainer CLI가 자동으로 HDF5 dataset을 선택하도록 바뀐 것은 아니다.
- MLX 학습용 loader는 별도 구현이 필요하다. 이 문서의 loader는 PyTorch `DataLoader` 기준이다.
