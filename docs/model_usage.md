# Model 사용 가이드

이 문서는 LALM 프로젝트의 모델 사용 방법을 설명합니다. 테스트 코드(`test/test_audio.py`, `test/test_LLM.py`)를 기준으로 작성되었습니다.

## 목차

1. [Audio Encoder](#audio-encoder)
   - [WhisperAudioEncoder](#whisperaudioencoder)
   - [HubertAudioEncoder](#hubertaudioencoder)
   - [Wav2Vec2AudioEncoder](#wav2vec2audioencoder)
2. [LLM Decoder](#llm-decoder)
3. [공통 기능](#공통-기능)

---

## Audio Encoder

### BaseAudioEncoder

모든 Audio Encoder의 기본 클래스입니다.

```python
from src.model.audio_encoder import BaseAudioEncoder

# 추상 클래스이므로 직접 사용하지 않고 하위 클래스를 사용합니다.
```

---

### WhisperAudioEncoder

Whisper 모델의 Encoder 부분만을 사용하는 Audio Encoder입니다.

#### 기본 초기화 (Frozen 모드)

```python
from src.model.audio_encoder import WhisperAudioEncoder

model = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",  # 또는 "openai/whisper-large-v3"
    finetune=False,
    use_lora=False,
    use_qlora=False,
    trust_remote_code=True
)

# 모든 파라미터가 frozen 상태 (requires_grad=False)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params:,}")  # 0
```

#### Finetune 모드

```python
model = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=True,  # 모든 파라미터가 trainable
    use_lora=False,
    use_qlora=False,
    trust_remote_code=True
)

# 모든 파라미터가 trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params:,}")  # > 0
```

#### LoRA 모드

```python
model = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
    use_lora=True,  # LoRA 활성화
    use_qlora=False,
    lora_r=8,          # LoRA rank
    lora_alpha=16,     # LoRA alpha
    lora_dropout=0.05, # LoRA dropout
    trust_remote_code=True
)

# LoRA 파라미터만 trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,}")
```

#### QLoRA 모드 (4-bit 양자화)

```python
model = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
    use_lora=False,
    use_qlora=True,  # QLoRA 활성화 (4-bit 양자화)
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    trust_remote_code=True
)

# QLoRA는 CUDA에 자동으로 로드될 수 있음
device = next(model.parameters()).device
print(f"Model device: {device}")
```

#### Forward Pass

```python
# Whisper는 mel spectrogram 입력 (batch_size, n_mels, seq_len)
n_mels = model.encoder.config.num_mel_bins  # 모델에 따라 다름 (보통 80)
batch_size, seq_len = 2, 3000
dummy_input = torch.randn(batch_size, n_mels, seq_len)

# Forward pass
hidden_states = model(dummy_input)

# 반환값: Tuple of all layer outputs
# 각 요소: (batch_size, seq_len, hidden_size)
# 마지막 요소가 최종 출력
print(f"Number of layers: {len(hidden_states)}")
print(f"Final layer shape: {hidden_states[-1].shape}")
```

#### Hidden Size 조회

```python
hidden_size = model.get_hidden_size()
print(f"Hidden size: {hidden_size}")
```

---

### HubertAudioEncoder

HuBERT 모델을 사용하는 Audio Encoder입니다. LoRA/QLoRA를 지원하지 않으며, Full fine-tuning만 지원합니다.

#### 기본 초기화 (Frozen 모드)

```python
from src.model.audio_encoder import HubertAudioEncoder

model = HubertAudioEncoder(
    model_name="facebook/hubert-base-ls960",
    finetune=False,
    trust_remote_code=True
)

# 모든 파라미터가 frozen
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params:,}")  # 0
```

#### Finetune 모드

```python
model = HubertAudioEncoder(
    model_name="facebook/hubert-base-ls960",
    finetune=True,  # 모든 파라미터 trainable
    trust_remote_code=True
)
```

#### Forward Pass

```python
# HuBERT는 raw audio 입력 (batch_size, seq_len)
# 예: 16kHz 샘플링 레이트에서 1초 = 16000 샘플
batch_size, seq_len = 2, 16000
dummy_input = torch.randn(batch_size, seq_len)

hidden_states = model(dummy_input)
# 반환값: Tuple of all layer outputs
print(f"Final layer shape: {hidden_states[-1].shape}")
```

---

### Wav2Vec2AudioEncoder

Wav2Vec2 모델을 사용하는 Audio Encoder입니다. LoRA/QLoRA를 지원하지 않으며, Full fine-tuning만 지원합니다.

#### 기본 초기화 (Frozen 모드)

```python
from src.model.audio_encoder import Wav2Vec2AudioEncoder

model = Wav2Vec2AudioEncoder(
    model_name="facebook/wav2vec2-base",
    finetune=False,
    trust_remote_code=True
)
```

#### Finetune 모드

```python
model = Wav2Vec2AudioEncoder(
    model_name="facebook/wav2vec2-base",
    finetune=True,
    trust_remote_code=True
)
```

#### Forward Pass

```python
# Wav2Vec2는 raw audio 입력 (batch_size, seq_len)
batch_size, seq_len = 2, 16000
dummy_input = torch.randn(batch_size, seq_len)

hidden_states = model(dummy_input)
print(f"Final layer shape: {hidden_states[-1].shape}")
```

---

## LLM Decoder

### LLMDecoder

다양한 LLM (Llama, Mistral, Qwen 등)을 지원하는 Decoder입니다. QLoRA/LoRA를 지원합니다.

#### 기본 초기화 (Frozen 모드)

```python
from src.model.LLM import LLMDecoder

model = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=False,
)

# 모든 파라미터가 frozen
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params:,}")  # 0
```

#### QLoRA 모드

```python
model = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=True,  # QLoRA 활성화
    use_lora=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# QLoRA 파라미터만 trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,}")
```

#### LoRA 모드

```python
model = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=True,  # LoRA 활성화
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)
```

#### Forward Pass

```python
# 텍스트를 tokenize
text = "Hello, how are you?"
inputs = model.tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Forward pass
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
print(f"Logits shape: {outputs.logits.shape}")

# Loss 계산 (labels 제공 시)
labels = input_ids.clone()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")
```

#### Text Generation

```python
model.eval()

prompt = "The capital of France is"
inputs = model.tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        do_sample=False,
        temperature=1.0,
    )

output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input: {prompt}")
print(f"Output: {output_text}")
```

#### Input Embeddings 조회

```python
embeddings = model.get_input_embeddings()
print(f"Embedding shape: {embeddings.weight.shape}")

# 임베딩 테스트
test_ids = torch.tensor([[1, 2, 3]])
embedded = embeddings(test_ids)
print(f"Embedded shape: {embedded.shape}")
```

#### Hidden Size 조회

```python
hidden_size = model.get_hidden_size()
print(f"Hidden size: {hidden_size}")
```

#### Tokenizer 사용

```python
text = "Hello, how are you?"

# Encode
encoded = model.tokenizer.encode(text)
print(f"Encoded: {encoded}")

# Decode
decoded = model.tokenizer.decode(encoded)
print(f"Decoded: {decoded}")

# Pad token 확인
print(f"Pad token: {model.tokenizer.pad_token}")
```

---

## 공통 기능

### 1. 학습 모드 / Inference 모드 설정

#### Audio Encoder

```python
# 학습 모드 설정
model.set_training_mode()
# - LoRA/QLoRA 사용 시: LoRA 파라미터만 trainable
# - Full fine-tuning 사용 시: 모든 파라미터 trainable

# Inference 모드 설정
model.set_inference_mode()
# - 모든 파라미터 frozen (requires_grad=False)
# - eval() 모드로 설정

# Inference 실행
model.set_inference_mode()
output = model.inference(input_data)  # torch.no_grad()로 실행
```

#### LLM Decoder

```python
# 학습 모드
model.train()

# Inference 모드
model.eval()

# Inference 실행
model.eval()
with torch.no_grad():
    output = model.generate(input_ids=input_ids, max_new_tokens=20)
```

### 2. LoRA 가중치 저장 및 불러오기

#### Audio Encoder

```python
import tempfile
import os

# 임시 디렉토리 생성
temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "lora_weights")

# LoRA 모델 생성
model = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
    use_lora=True,
    use_qlora=False,
    lora_r=8,
    lora_alpha=16,
)

# 학습 (예시)
model.set_training_mode()
# ... 학습 코드 ...

# LoRA 가중치 저장
model.save_lora_weights(save_path)

# 새 모델에 LoRA 가중치 불러오기
model_new = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
    use_lora=True,
    use_qlora=False,
    lora_r=8,
    lora_alpha=16,
)
model_new.load_lora_weights(save_path)
```

#### LLM Decoder

```python
import tempfile
import os

temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "lora_weights")

# LoRA 모델 생성
model = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)

# 학습 (예시)
model.train()
# ... 학습 코드 ...

# LoRA 가중치 저장
model.save_lora_weights(save_path)

# 새 모델에 LoRA 가중치 불러오기
model_new = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)
model_new.load_lora_weights(save_path)
```

### 3. 전체 모델 가중치 저장 및 불러오기 (Audio Encoder만)

```python
import tempfile
import os

temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "model_weights")

# 모델 생성
model = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=True,
)

# 학습
model.set_training_mode()
# ... 학습 코드 ...

# 전체 모델 가중치 저장
model.save_model_weights(save_path)

# 새 모델에 가중치 불러오기
model_new = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
)
model_new.load_model_weights(save_path)
```

### 4. Continuous Learning (연속 학습)

#### Audio Encoder

```python
# 1단계: 초기 학습 및 저장
model1 = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
    use_lora=True,
    use_qlora=False,
    lora_r=8,
    lora_alpha=16,
)

model1.set_training_mode()
# ... 첫 번째 학습 ...

model1.save_lora_weights("checkpoint_1")

# 2단계: 저장된 가중치 불러와서 추가 학습
model2 = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
    use_lora=True,
    use_qlora=False,
    lora_r=8,
    lora_alpha=16,
)

model2.load_lora_weights("checkpoint_1")
model2.set_training_mode()

# ... 추가 학습 ...
model2.save_lora_weights("checkpoint_2")
```

#### LLM Decoder

```python
# 1단계: 초기 학습 및 저장
model1 = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)

model1.train()
# ... 첫 번째 학습 ...

model1.save_lora_weights("checkpoint_1")

# 2단계: 저장된 가중치 불러와서 추가 학습
model2 = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)

model2.load_lora_weights("checkpoint_1")
model2.train()

# ... 추가 학습 ...
model2.save_lora_weights("checkpoint_2")
```

### 5. 저장된 가중치로 Inference

#### Audio Encoder

```python
# 학습된 모델 저장
model_train = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=True,
)
model_train.set_training_mode()
# ... 학습 ...
model_train.save_model_weights("trained_model")

# Inference 모드로 불러오기
model_inference = WhisperAudioEncoder(
    model_name="/path/to/whisper-large-v3",
    finetune=False,
)
model_inference.load_model_weights("trained_model")
model_inference.set_inference_mode()

# Inference 실행
input_data = torch.randn(1, 80, 3000)  # 예시
output = model_inference.inference(input_data)
```

#### LLM Decoder

```python
# 학습된 LoRA 모델 저장
model_train = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)
model_train.train()
# ... 학습 ...
model_train.save_lora_weights("trained_lora")

# Inference 모드로 불러오기
model_inference = LLMDecoder(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_qlora=False,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)
model_inference.load_lora_weights("trained_lora")
model_inference.eval()

# Inference 실행
prompt = "The capital of France is"
inputs = model_inference.tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.no_grad():
    output_ids = model_inference.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        do_sample=False,
    )

output_text = model_inference.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

---

## 주요 팁

1. **디바이스 관리**: QLoRA를 사용하면 모델이 자동으로 CUDA에 로드될 수 있습니다. 입력 데이터도 같은 디바이스로 이동해야 합니다.

```python
device = next(model.parameters()).device
input_data = input_data.to(device)
```

2. **메모리 효율성**: 
   - QLoRA는 4-bit 양자화를 사용하여 메모리 사용량을 크게 줄입니다.
   - LoRA는 일부 파라미터만 학습하므로 메모리 효율적입니다.
   - Full fine-tuning은 가장 많은 메모리가 필요합니다.

3. **학습 모드 전환**:
   - Audio Encoder: `set_training_mode()` / `set_inference_mode()` 사용
   - LLM Decoder: `train()` / `eval()` 사용

4. **가중치 저장**:
   - LoRA 가중치만 저장하면 용량이 매우 작습니다.
   - 전체 모델 가중치 저장은 용량이 큽니다 (Audio Encoder만 지원).

---

## 참고

- 테스트 코드: `test/test_audio.py`, `test/test_LLM.py`
- 모델 구현: `src/model/audio_encoder.py`, `src/model/LLM.py`

