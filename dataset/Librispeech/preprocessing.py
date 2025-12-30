import os
import json
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

# LibriSpeech 데이터셋 경로 (상대경로 또는 환경변수 사용)
# 환경변수가 설정되어 있으면 사용하고, 없으면 현재 스크립트 위치 기준 상대경로 사용
LIBRISPEECH_ROOT = os.getenv("LIBRISPEECH_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "data", "LibriSpeech"))

# 출력 파일 경로
OUTPUT_JSONL_TRAIN = "librispeech_train.jsonl"
OUTPUT_JSONL_DEV = "librispeech_dev.jsonl"
OUTPUT_JSONL_TEST = "librispeech_test.jsonl"

# LibriSpeech 데이터셋 구조 파싱
def parse_librispeech(root_dir, split_dirs):
    """LibriSpeech 데이터셋을 JSONL 형식으로 변환"""
    data = []
    
    for split_dir in split_dirs:
        split_path = Path(root_dir) / split_dir
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist, skipping...")
            continue
            
        # 각 화자 디렉토리 순회
        for speaker_dir in tqdm(split_path.iterdir(), desc=f"Processing {split_dir}"):
            if not speaker_dir.is_dir():
                continue
                
            # 각 챕터 디렉토리 순회
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                    
                # .trans.txt 파일 읽기 (형식: speaker_id-chapter_id.trans.txt)
                speaker_id = speaker_dir.name
                chapter_id = chapter_dir.name
                trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                if not trans_file.exists():
                    continue
                    
                # 트랜스크립트 로드
                with open(trans_file, 'r') as f:
                    transcripts = {}
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]
                
                # 오디오 파일 처리
                for audio_file in chapter_dir.glob("*.flac"):
                    audio_id = audio_file.stem
                    if audio_id in transcripts:
                        # 오디오 파일 정보 추출
                        info = sf.info(str(audio_file))
                        
                        # 상대경로로 변환 (root_dir 기준)
                        audio_path = Path(audio_file)
                        root_path = Path(root_dir).resolve()
                        relative_audio_path = audio_path.relative_to(root_path)
                        
                        data.append({
                            "audio": str(relative_audio_path),
                            "text": transcripts[audio_id],
                            "duration": info.duration,
                            "sample_rate": info.samplerate
                        })
    
    return data

# Train 데이터 파싱
print("Parsing LibriSpeech train dataset...")
train_dirs = ["train-clean-100", "train-clean-360", "train-other-500"]
train_data = parse_librispeech(LIBRISPEECH_ROOT, train_dirs)

# Dev 데이터 파싱
print("\nParsing LibriSpeech dev dataset...")
dev_dirs = ["dev-clean", "dev-other"]
dev_data = parse_librispeech(LIBRISPEECH_ROOT, dev_dirs)

# Test 데이터 파싱
print("\nParsing LibriSpeech test dataset...")
test_dirs = ["test-clean", "test-other"]
test_data = parse_librispeech(LIBRISPEECH_ROOT, test_dirs)

# JSONL 파일로 저장
print(f"\nSaving {len(train_data)} train samples to {OUTPUT_JSONL_TRAIN}...")
with open(OUTPUT_JSONL_TRAIN, 'w', encoding='utf-8') as f:
    for item in tqdm(train_data, desc="Writing train JSONL"):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Saving {len(dev_data)} dev samples to {OUTPUT_JSONL_DEV}...")
with open(OUTPUT_JSONL_DEV, 'w', encoding='utf-8') as f:
    for item in tqdm(dev_data, desc="Writing dev JSONL"):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Saving {len(test_data)} test samples to {OUTPUT_JSONL_TEST}...")
with open(OUTPUT_JSONL_TEST, 'w', encoding='utf-8') as f:
    for item in tqdm(test_data, desc="Writing test JSONL"):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nDone!")
print(f"Train: {len(train_data)} samples saved to {OUTPUT_JSONL_TRAIN}")
print(f"Dev: {len(dev_data)} samples saved to {OUTPUT_JSONL_DEV}")
print(f"Test: {len(test_data)} samples saved to {OUTPUT_JSONL_TEST}")
