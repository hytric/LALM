# LALM

Large Language Model for Audio-to-Text Conversion

## 프로젝트 개요

LALM은 오디오를 텍스트로 변환하는 멀티모달 모델입니다. 오디오 인코더와 대규모 언어 모델(LLM)을 결합하여 음성 인식 및 다양한 오디오 기반 태스크를 수행합니다.

## 모델 구조

### Audio Encoder
- **HuBERT**: Self-supervised learning 기반 오디오 인코더
- **Whisper**: OpenAI의 음성 인식 모델 (Encoder 부분만 사용)
- **Wav2Vec2**: Wav2Vec2 기반 오디오 인코더

각 인코더는 Full fine-tuning, LoRA, QLoRA 방식을 지원합니다.

### Adapter
Audio Encoder의 출력을 LLM 입력 차원으로 변환하는 프로젝션 레이어입니다.
- 2개의 Linear layer
- 1개의 GeLU 활성화 함수

### LLM Decoder
- **지원 모델**: Llama, Mistral, Qwen 등 다양한 LLM
- **최적화**: QLoRA (4-bit 양자화), LoRA 지원
- **기능**: 텍스트 생성, 음성 인식, 다중 태스크 처리

## 학습 전략

### Stage 1: ASR (Automatic Speech Recognition)
- 음성 인식 태스크에 집중하여 초기 학습
- 오디오-텍스트 매핑 학습

### Stage 2: Multi-task Learning
- ASR 태스크 유지
- 감정 인식, 화자 인식 등 추가 태스크 병합 학습

## 데이터셋

- **Speechcraft**: 음성-텍스트 변환 데이터셋
- **IEMOCAP**: 감정 인식 데이터셋
- **MELD**: 멀티모달 감정 인식 데이터셋

## 주요 기능

- 다양한 오디오 인코더 지원 (HuBERT, Whisper, Wav2Vec2)
- 메모리 효율적인 학습 (QLoRA, LoRA)
- Continuous Learning 지원 (체크포인트 저장 및 불러오기)
- Flexible 학습 모드 (Full fine-tuning, LoRA, QLoRA, Frozen)


