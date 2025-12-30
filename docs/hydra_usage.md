# Hydra 사용법 가이드

이 문서는 LALM 프로젝트에서 Hydra를 사용하는 방법을 설명합니다.

## 목차

1. [Hydra란?](#hydra란)
2. [설치](#설치)
3. [코드 구성 방법](#코드-구성-방법)
4. [Config 구조](#config-구조)
5. [기본 사용법](#기본-사용법)
6. [Config Group 변경](#config-group-변경)
7. [설정 오버라이드](#설정-오버라이드)
8. [고급 사용법](#고급-사용법)
9. [Config 파일 구조](#config-파일-구조)
10. [문제 해결](#문제-해결)

## Hydra란?

Hydra는 Facebook에서 개발한 Python 설정 관리 프레임워크입니다. 복잡한 실험 설정을 쉽게 관리하고, command-line에서 설정을 변경할 수 있게 해줍니다.

### 주요 장점

- **Config Composition**: 여러 config 파일을 조합하여 사용
- **Command-line Override**: 실행 시 설정 변경 가능
- **Config 검증**: 설정 오류를 사전에 발견
- **실험 관리**: 각 실행마다 config를 자동으로 저장

## 설치

```bash
pip install hydra-core
```

또는 requirements.txt에 추가:

```txt
hydra-core>=1.3.0
```

## 코드 구성 방법

이 섹션에서는 Hydra가 프로젝트 코드에 어떻게 통합되어 있는지 설명합니다.

### 프로젝트 구조

```
LALM/
├── main.py                    # Hydra 진입점
├── config/                     # Hydra config 파일들
│   ├── config.yaml
│   ├── dataset/
│   ├── model/
│   ├── trainer/
│   └── inference/
├── src/
│   ├── train.py               # 학습 메인 로직
│   ├── trainer.py             # PyTorch Lightning 모듈
│   ├── utils/
│   │   └── common_utils.py   # Config 인스턴스화 유틸리티
│   └── ...
└── ...
```

### 1. 진입점: `main.py`

Hydra 데코레이터를 사용하여 진입점을 설정합니다:

```python
from src.train import main
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_with_hydra(cfg: DictConfig):
    """
    Main entry point with Hydra.
    
    Args:
        cfg: Hydra DictConfig 객체
    """
    main(cfg)

if __name__ == "__main__":
    run_with_hydra()
```

**주요 포인트:**
- `@hydra.main` 데코레이터가 Hydra를 초기화하고 config를 로드
- `config_path`: config 파일들이 있는 디렉토리 경로
- `config_name`: 기본으로 사용할 config 파일 이름 (확장자 제외)
- 함수는 `DictConfig` 타입의 `cfg` 파라미터를 받음

### 2. 학습 로직: `src/train.py`

Hydra에서 받은 config를 처리하는 메인 함수:

```python
from omegaconf import DictConfig, OmegaConf
from src.utils.common_utils import instantiate_from_config

def _convert_to_dict(cfg):
    """Convert DictConfig to dict if needed"""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg

def main(cfg: DictConfig):
    """
    Main training function using Hydra config.
    
    Args:
        cfg: Hydra DictConfig containing all configuration
    """
    # DictConfig를 dict로 변환 (기존 코드와 호환성)
    config = _convert_to_dict(cfg)
    
    # Config에서 설정 추출
    experiment_config = config.get('experiment', {})
    model_config = config['model']
    data_config = config.get('data', {})
    trainer_config = config.get('trainer', {})
    
    # ... 학습 로직 ...
```

**주요 포인트:**
- `DictConfig`를 dict로 변환하여 기존 코드와 호환성 유지
- Config에서 필요한 섹션을 추출하여 사용
- `instantiate_from_config()` 함수로 config에서 객체 생성

### 3. Config 인스턴스화: `src/utils/common_utils.py`

Config에서 Python 객체를 생성하는 핵심 함수:

```python
from omegaconf import DictConfig, OmegaConf

def _convert_to_dict(config):
    """
    Convert Hydra DictConfig to regular dict if needed.
    """
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return config

def instantiate_from_config(config):
    """
    Instantiate class from config dictionary.
    Supports nested configs by recursively processing params.
    Also supports Hydra DictConfig.
    
    Args:
        config: dict or DictConfig containing 'target' or 'class_name' key
    
    Example config:
        {
            "target": "src.model.LALM.LALMModel",
            "params": {
                "audio_encoder": {
                    "target": "src.model.audio_encoder.WhisperAudioEncoder",
                    "params": {...}
                }
            }
        }
    """
    # DictConfig를 dict로 변환
    config = _convert_to_dict(config)
    
    # 클래스 이름 추출
    class_name = config.pop("target", None) or config.pop("class_name", None)
    if class_name is None:
        raise ValueError("Config must contain 'target' or 'class_name' key")
    
    # 동적으로 클래스 가져오기
    cls = get_obj_from_str(class_name)
    
    # 파라미터 추출
    if "params" in config:
        params = _convert_to_dict(config["params"]).copy()
    else:
        params = config.copy()
    
    # 중첩된 config 재귀적으로 처리
    for key, value in params.items():
        if isinstance(value, dict) and ("target" in value or "class_name" in value):
            params[key] = instantiate_from_config(value)
    
    # 클래스 인스턴스 생성
    return cls(**params)
```

**주요 포인트:**
- `target` 필드에 클래스의 전체 경로 지정 (예: `src.model.LALM.LALMModel`)
- `params` 필드에 생성자에 전달할 파라미터 지정
- 중첩된 config도 재귀적으로 처리하여 복잡한 객체 구조 생성 가능
- Hydra `DictConfig`와 일반 `dict` 모두 지원

### 4. Config 파일 구조

Config 파일은 다음과 같은 구조를 가집니다:

```yaml
# config/model/qlora.yaml
base_learning_rate: 1.0e-4
scale_lr: False
target: src.model.LALM.LALMModel
params:
  audio_encoder:
    target: src.model.audio_encoder.WhisperAudioEncoder
    params:
      model_name: "/path/to/model"
      use_qlora: True
  llm_decoder:
    target: src.model.LLM.LLMDecoder
    params:
      model_name: "meta-llama/Llama-3.2-3B-Instruct"
      use_qlora: True
      lora_r: 16
```

**Config 필드 설명:**
- `target`: 인스턴스화할 클래스의 전체 경로
- `params`: 클래스 생성자에 전달할 파라미터들
- 중첩 구조: `params` 내부에 또 다른 `target`과 `params`를 가진 dict를 넣으면 재귀적으로 처리

### 5. 실행 흐름

```
1. python main.py 실행
   ↓
2. @hydra.main 데코레이터가 config 로드
   - config/config.yaml 읽기
   - defaults에 따라 dataset, model, trainer config 조합
   ↓
3. run_with_hydra(cfg: DictConfig) 호출
   ↓
4. src.train.main(cfg) 호출
   - DictConfig를 dict로 변환
   - Config에서 각 섹션 추출
   ↓
5. instantiate_from_config()로 객체 생성
   - Model, Dataset, Trainer 등 생성
   ↓
6. 학습 시작
```

### 6. 새로운 Config Group 추가하기

새로운 config group을 추가하는 방법:

#### Step 1: Config 파일 생성

```yaml
# config/model/my_custom_model.yaml
base_learning_rate: 2.0e-4
scale_lr: True
target: src.model.LALM.LALMModel
params:
  audio_encoder:
    target: src.model.audio_encoder.WhisperAudioEncoder
    params:
      model_name: "/custom/path/to/model"
      use_qlora: False
  llm_decoder:
    target: src.model.LLM.LLMDecoder
    params:
      model_name: "meta-llama/Llama-3.2-1B-Instruct"
      use_qlora: False
```

#### Step 2: 사용

```bash
python main.py model=my_custom_model
```

### 7. 새로운 Dataset 추가하기

#### Step 1: Dataset 클래스 구현

```python
# src/data/my_dataset.py
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        # ...
    
    def __getitem__(self, idx):
        # ...
```

#### Step 2: Config 파일 생성

```yaml
# config/dataset/my_dataset.yaml
target: src.data.my_dataset.MyDataset
params:
  data_path: "dataset/my_data"
  batch_size: 8
  num_workers: 4
  train:
    target: src.data.my_dataset.MyDataset
    params:
      data_path: "dataset/my_data/train"
  validation:
    target: src.data.my_dataset.MyDataset
    params:
      data_path: "dataset/my_data/val"
```

#### Step 3: 사용

```bash
python main.py dataset=my_dataset
```

### 8. Config에서 함수/메서드 호출하기

Config에서 함수나 메서드를 직접 호출할 수는 없지만, 팩토리 패턴을 사용할 수 있습니다:

```python
# src/utils/factory.py
def create_optimizer(model_params, lr, optimizer_type="adam"):
    if optimizer_type == "adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model_params, lr=lr)
    # ...
```

```yaml
# config/optimizer/adam.yaml
target: src.utils.factory.create_optimizer
params:
  optimizer_type: "adam"
  lr: 1.0e-4
```

### 9. Best Practices

1. **Config 파일 명명 규칙**
   - 소문자와 언더스코어 사용: `my_config.yaml`
   - 의미 있는 이름 사용: `librispeech_train.yaml` (O), `config1.yaml` (X)

2. **Config 구조화**
   - 관련된 설정은 그룹으로 묶기
   - 중복을 피하고 defaults 활용
   - 주석으로 각 설정의 의미 설명

3. **타입 안정성**
   - 가능하면 명시적인 타입 지정
   - Config 검증 로직 추가 고려

4. **환경별 설정**
   - 개발/프로덕션 환경별로 다른 config 사용
   - 민감한 정보는 환경변수로 관리

5. **Config 버전 관리**
   - Config 파일도 코드처럼 버전 관리
   - 변경 이력을 명확히 기록

## Config 구조

프로젝트의 config는 다음과 같은 구조로 구성되어 있습니다:

```
config/
├── config.yaml              # 메인 config (defaults 정의)
├── dataset/                  # 데이터셋 config group
│   ├── librispeech_train.yaml
│   ├── librispeech_inf.yaml
│   ├── speechcraft_train.yaml
│   └── speechcraft_inf.yaml
├── model/                    # 모델 config group
│   ├── base.yaml
│   └── qlora.yaml
├── trainer/                  # 트레이너 config group
│   └── base.yaml
└── inference/                # 추론 config group
    └── base.yaml
```

## 기본 사용법

### 1. 기본 실행

기본 설정으로 학습을 시작합니다:

```bash
python main.py
```

이 명령은 다음 설정을 사용합니다:
- `dataset: librispeech_train`
- `model: qlora`
- `trainer: base`

### 2. Config 확인

Hydra는 실행 시 현재 설정을 출력합니다. 또한 `outputs/` 디렉토리에 config 파일을 저장합니다.

## Config Group 변경

### Dataset 변경

LibriSpeech 대신 SpeechCraft 데이터셋을 사용:

```bash
python main.py dataset=speechcraft_train
```

### Model 변경

QLoRA 대신 기본 모델을 사용:

```bash
python main.py model=base
```

### 여러 Group 동시 변경

```bash
python main.py dataset=speechcraft_train model=base
```

## 설정 오버라이드

### 1. Experiment 설정 변경

```bash
# Experiment 이름 설정
python main.py experiment.name=my_experiment

# Logger 변경 (tensorboard → wandb)
python main.py experiment.logger=wandb

# 로그 저장 디렉토리 변경
python main.py experiment.save_dir=./my_logs
```

### 2. Model 설정 변경

```bash
# Learning rate 변경
python main.py model.base_learning_rate=2.0e-4

# LLM 모델 변경
python main.py model.params.llm_decoder.params.model_name=meta-llama/Llama-3.2-1B-Instruct

# Audio encoder 모델 변경
python main.py model.params.audio_encoder.params.model_name=/path/to/whisper-model
```

### 3. Trainer 설정 변경

```bash
# 최대 epoch 수 변경
python main.py trainer.trainer.max_epochs=200

# Batch size 변경
python main.py trainer.trainer.batch_size=8

# Precision 변경 (16-mixed → 32-true)
python main.py trainer.trainer.precision=32-true

# Checkpoint에서 재개
python main.py trainer.trainer.resume_from_checkpoint=./logs/experiment/checkpoints/epoch=010-step=1000.ckpt

# GPU 개수 변경
python main.py trainer.trainer.devices=2

# Gradient accumulation 변경
python main.py trainer.trainer.accumulate_grad_batches=4
```

### 4. Dataset 설정 변경

```bash
# LibriSpeech root 경로 설정
python main.py dataset.params.train.params.librispeech_root=/path/to/librispeech

# JSONL 파일 경로 변경
python main.py dataset.params.train.params.jsonl_file=dataset/Librispeech/custom_train.jsonl

# Batch size 변경
python main.py dataset.params.batch_size=8

# Num workers 변경
python main.py dataset.params.num_workers=8
```

### 5. 중첩된 설정 변경

점(`.`) 표기법으로 중첩된 설정을 변경할 수 있습니다:

```bash
# LoRA 파라미터 변경
python main.py model.params.llm_decoder.params.lora_r=32
python main.py model.params.llm_decoder.params.lora_alpha=64
python main.py model.params.llm_decoder.params.lora_dropout=0.1
```

## 고급 사용법

### 1. Config 파일 직접 지정

특정 config 파일을 직접 지정:

```bash
python main.py --config-path=config --config-name=config dataset=librispeech_train
```

### 2. Config 출력 및 검증

Config를 출력만 하고 실행하지 않기:

```bash
python main.py --cfg job
```

### 3. Multi-run (Sweep)

여러 설정으로 실험을 자동 실행:

```bash
# Learning rate sweep
python main.py -m model.base_learning_rate=1e-4,2e-4,5e-4

# Dataset과 model 조합
python main.py -m dataset=librispeech_train,speechcraft_train model=base,qlora

# 여러 파라미터 조합
python main.py -m \
  dataset=librispeech_train,speechcraft_train \
  trainer.trainer.max_epochs=50,100 \
  trainer.trainer.batch_size=4,8
```

### 4. Config 파일 저장 위치 변경

기본적으로 Hydra는 `outputs/` 디렉토리에 config를 저장합니다. 위치를 변경:

```bash
python main.py hydra.run.dir=./custom_output
```

### 5. Config 파일 편집

Hydra가 생성한 config 파일을 편집하여 재사용:

```bash
# 1. 기본 실행하여 config 생성
python main.py experiment.name=test_run

# 2. outputs/test_run/YYYY-MM-DD_HH-MM-SS/.hydra/config.yaml 파일 편집

# 3. 편집된 config로 실행
python main.py --config-path=outputs/test_run/YYYY-MM-DD_HH-MM-SS/.hydra --config-name=config
```

## Config 파일 구조

### 1. 메인 Config (`config/config.yaml`)

```yaml
defaults:
  - dataset: librispeech_train
  - model: qlora
  - trainer: base
  - _self_

experiment:
  name: null
  save_dir: ./logs
  logger: tensorboard
```

- `defaults`: 사용할 config group들을 지정
- `_self_`: 현재 파일의 설정이 다른 config 위에 적용됨을 의미

### 2. Dataset Config 예시

```yaml
# config/dataset/librispeech_train.yaml
target: src.data.librispeech.LibriSpeechDataset
params:
  data_root: null
  batch_size: 4
  num_workers: 4
  train:
    target: src.data.librispeech.LibriSpeechDataset
    params:
      jsonl_file: "dataset/Librispeech/librispeech_train.jsonl"
      librispeech_root: null
      stage: 1
  validation:
    target: src.data.librispeech.LibriSpeechDataset
    params:
      jsonl_file: "dataset/Librispeech/librispeech_dev.jsonl"
      librispeech_root: null
      stage: 1
```

### 3. Model Config 예시

```yaml
# config/model/qlora.yaml
base_learning_rate: 1.0e-4
scale_lr: False
target: src.model.LALM.LALMModel
params:
  audio_encoder:
    target: src.model.audio_encoder.WhisperAudioEncoder
    params:
      model_name: "/home/jhkim/jhkim/model/whisper-large-v3"
      use_qlora: True
  llm_decoder:
    target: src.model.LLM.LLMDecoder
    params:
      model_name: "meta-llama/Llama-3.2-3B-Instruct"
      use_qlora: True
      lora_r: 16
      lora_alpha: 32
      lora_dropout: 0.05
```

### 4. Trainer Config 예시

```yaml
# config/trainer/base.yaml
callbacks:
  model_checkpoint:
    target: pytorch_lightning.callbacks.ModelCheckpoint
    params:
      filename: "{epoch:04}-{step:06}"
      every_n_epochs: 1
      save_top_k: 3
      monitor: "val/loss"
      mode: "min"

trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: 1
  precision: 16-mixed
  batch_size: 4
  num_workers: 4
  resume_from_checkpoint: null
```

## 사용 예시

### 예시 1: 기본 학습

```bash
python main.py
```

### 예시 2: SpeechCraft 데이터셋으로 학습

```bash
python main.py dataset=speechcraft_train
```

### 예시 3: 커스텀 실험 설정

```bash
python main.py \
  dataset=librispeech_train \
  model=base \
  experiment.name=librispeech_base_experiment \
  experiment.logger=wandb \
  trainer.trainer.max_epochs=200 \
  trainer.trainer.batch_size=8
```

### 예시 4: Checkpoint에서 재개

```bash
python main.py \
  trainer.trainer.resume_from_checkpoint=./logs/my_experiment/checkpoints/epoch=050-step=5000.ckpt \
  experiment.name=resumed_experiment
```

### 예시 5: Hyperparameter Sweep

```bash
python main.py -m \
  model.base_learning_rate=1e-4,2e-4,5e-4 \
  trainer.trainer.batch_size=4,8,16
```

이 명령은 3 × 3 = 9개의 실험을 자동으로 실행합니다.

### 예시 6: LoRA 파라미터 튜닝

```bash
python main.py -m \
  model.params.llm_decoder.params.lora_r=8,16,32 \
  model.params.llm_decoder.params.lora_alpha=16,32,64
```

## 문제 해결

### 1. Config 파일을 찾을 수 없음

```
Could not find 'dataset/xxx'
```

**해결**: `config/dataset/` 디렉토리에 해당 파일이 있는지 확인하세요.

### 2. 설정 오버라이드가 적용되지 않음

**원인**: 잘못된 경로를 사용했을 수 있습니다.

**해결**: Config 구조를 확인하고 올바른 경로를 사용하세요.

```bash
# 잘못된 예
python main.py model.lora_r=32

# 올바른 예
python main.py model.params.llm_decoder.params.lora_r=32
```

### 3. Hydra 출력 디렉토리 정리

Hydra는 기본적으로 `outputs/` 디렉토리에 실행 결과를 저장합니다. 정리하려면:

```bash
rm -rf outputs/
```

또는 `.gitignore`에 추가:

```gitignore
outputs/
.hydra/
```

### 4. Config 검증

Config 파일의 문법 오류를 확인:

```bash
python main.py --cfg job --dry-run
```

## 추가 리소스

- [Hydra 공식 문서](https://hydra.cc/)
- [Hydra 튜토리얼](https://hydra.cc/docs/tutorials/intro/)
- [Config Groups 가이드](https://hydra.cc/docs/tutorials/structured_config/config_groups/)

## Tips

1. **Config 파일 명명 규칙**: 소문자와 언더스코어 사용 (`librispeech_train.yaml`)

2. **설정 우선순위**: Command-line override > Config file > Defaults

3. **Config 재사용**: 자주 사용하는 설정 조합은 별도 config 파일로 만들 수 있습니다

4. **환경변수 사용**: 민감한 정보(API 키 등)는 환경변수로 관리:

```yaml
model_name: ${oc.env:MODEL_PATH,/default/path}
```

5. **Config 검증**: `instantiate_from_config()` 함수가 자동으로 config를 검증합니다

