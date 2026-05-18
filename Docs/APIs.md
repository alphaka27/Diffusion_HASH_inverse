# Function Call structure

Execute hash_main.py (Main EntryPoint)  
&nbsp;&nbsp;&nbsp;&nbsp; $\rightarrow$ Call each hash algorithm's main function

## Hash algorithm's Structure
__Implement using class syntax__  
### Input & Output
__Input__: Bytes  
__Output__: Bytes | String

Class name is must be __UPPER CASE__  
```__init__``` function  
```python
def __init__(self, hash_config, steplogs, is_verbose=True)
```


__Hash algorithm's main function__
```python
def digest(self, message = None, message_len = -1)
```

## Random Character/Bit Generator
__Implement using class syntax__  
### Input & Output
__Input__: None  
__Output__: Bytes

## HDF5 Tensor Dataset API

JSON log에서 생성한 이미지를 HDF5 tensor shard로 저장하고 PyTorch에서 바로 읽을 때 사용한다. 자세한 예제와 shard 구조는 `Docs/HDF5TensorDataset.md`를 기준으로 한다.

### HDF5Maker

```python
from diffusion_hash_inv.utils import HDF5Maker

hdf5_maker = HDF5Maker(runtime_cfg, io_controller)
hdf5_maker.main(
    logs=log_paths,
    rgb_config=rgb_config,
    workers=4,
    output_name="hash_tensors",
    shard_size=1000,
    include_paths=None,
    channels=3,
    normalize=False,
    compression="gzip",
    preserve_log_hierarchy=True,
)
```

__Input__: JSON log paths
__Output__: `data/tensor_datasets/<output_name>/manifest.json` and `*.h5` shard files. Each shard keeps the flat `records` index and, when `preserve_log_hierarchy=True`, a `logs/<source_log>/Message/Hex` and `logs/<source_log>/Logs/...` hierarchy.

### HDF5TensorDataset

```python
from diffusion_hash_inv.utils import HDF5TensorDataset

dataset = HDF5TensorDataset(
    "data/tensor_datasets/hash_tensors",
    include_paths=("message.png",),
    require_same_shape=True,
)

tensor, index = dataset[0]
metadata = dataset.metadata(0)
```

__Input__: HDF5 tensor dataset root or `manifest.json`
__Output__: `(tensor, index_tensor)` by default

### create_hdf5_tensor_dataloader

```python
from diffusion_hash_inv.utils import create_hdf5_tensor_dataloader

loader = create_hdf5_tensor_dataloader(
    "data/tensor_datasets/hash_tensors",
    batch_size=64,
    shuffle=True,
    num_workers=4,
    include_paths=("message.png",),
)
```

PyTorch 기본 collate는 같은 batch 안의 tensor shape가 모두 같아야 한다. 학습용으로는 `include_paths=("message.png",)`처럼 동일 image type만 선택한다.
