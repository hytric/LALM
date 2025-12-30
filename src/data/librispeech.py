import json
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer

def preprocess_text(text):
    """Preprocess text for downstream tasks"""
    if text is None or text == "":
        return ""
    # Convert to lowercase
    text = text.lower()
    # Basic punctuation normalization
    text = text.replace('<period>', '.').replace('<comma>', ',')
    return text.strip()

def get_instruction_and_target(row, task):
    """
    Get instruction and target for a specific task
    
    Args:
        row: Dictionary with all metadata
        task: "asr", "sed", or "ser"
    
    Returns:
        instruction: str, target: str
    """
    if task == "asr":
        return "Transcribe: ", preprocess_text(row.get('Transcript', row.get('text', '')))
    elif task == "sed":
        return row.get('Ins', 'Describe the emotional characteristics of this speech: '), preprocess_text(row.get('Des', ''))
    elif task == "ser":
        return "Classify emotion: ", row.get('Emotion', '').lower() if row.get('Emotion') else ''
    else:
        raise ValueError(f"Unknown task: {task}")

class LibriSpeechDataset(Dataset):
    def __init__(self, jsonl_file, librispeech_root=None, stage=1):
        """
        Dataset for LibriSpeech JSONL format.
        
        Args:
            jsonl_file: Path to JSONL file (e.g., librispeech_dev.jsonl)
            librispeech_root: Root directory of LibriSpeech dataset. 
                            If None, uses LIBRISPEECH_ROOT env var or default relative path.
            stage: Training stage (1 for ASR only, 2 for multi-task)
        
        Each line in JSONL should have:
            {
                "audio": "path/to/audio.flac" (relative or absolute),
                "text": "transcript text",
                "duration": float,
                "sample_rate": int
            }
        """
        self.stage = stage
        self.jsonl_file = jsonl_file
        
        # Set LibriSpeech root directory
        if librispeech_root is None:
            librispeech_root = os.getenv("LIBRISPEECH_ROOT")
            if librispeech_root is None:
                # Default: assume JSONL is in dataset/Librispeech/, data is in ../../data/LibriSpeech
                jsonl_dir = Path(jsonl_file).parent
                librispeech_root = os.path.join(jsonl_dir, "..", "..", "data", "LibriSpeech")
                librispeech_root = os.path.normpath(librispeech_root)
        self.librispeech_root = Path(librispeech_root) if librispeech_root else None
        
        # Load JSONL data
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        if self.stage == 1:
            self.labels = ['sid', 'Transcript']
        elif self.stage == 2:
            self.labels = ['sid', 'Gender', 'Age', 'Speed', 'Pitch', 'Energy', 'Emotion', 'Category', 'Transcript', 'Des', 'Ins']
    
    def _resolve_audio_path(self, audio_path):
        """Resolve audio path (handle both absolute and relative paths)"""
        audio_path = str(audio_path)
        
        # If absolute path exists, use it
        if os.path.isabs(audio_path) and os.path.exists(audio_path):
            return audio_path
        
        # If relative path, resolve against librispeech_root
        if self.librispeech_root:
            resolved_path = self.librispeech_root / audio_path
            if resolved_path.exists():
                return str(resolved_path)
        
        # Fallback: try as-is
        if os.path.exists(audio_path):
            return audio_path
        
        # Last resort: try relative to JSONL file directory
        jsonl_dir = Path(self.jsonl_file).parent
        fallback_path = jsonl_dir / audio_path
        if fallback_path.exists():
            return str(fallback_path)
        
        return audio_path
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio file"""
        resolved_path = self._resolve_audio_path(audio_path)
        waveform, sample_rate = torchaudio.load(resolved_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        
        # Prepare input
        waveform_np = waveform.squeeze().numpy()
        return torch.from_numpy(waveform_np)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio']
        audio = self._load_audio(audio_path)
        
        # Create metadata tuple compatible with existing collator
        if self.stage == 1:
            # Stage 1: ASR only - use index as sid
            sid = idx
            transcript = item.get('text', '')
            metadata = (sid, transcript)
        elif self.stage == 2:
            # Stage 2: Multi-task - LibriSpeech doesn't have these, use defaults
            sid = idx
            transcript = item.get('text', '')
            metadata = (sid, None, None, None, None, None, None, None, transcript, None, None)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        return audio, metadata


class LibriSpeechCollator:
    """Collator for batching LibriSpeech data with instructions and targets"""
    def __init__(self, model_name, stage=1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.stage = stage
    
    def __call__(self, batch):
        """
        Collate batch of (waveform, metadata) tuples

        Returns:
            waveform: torch.Tensor [B, T] - padded audio waveform
            pre_tokenized_ids: torch.Tensor [B, L1] - instruction tokens
            post_tokenized_ids: torch.Tensor [B, L2] - separator tokens (empty)
            output_tokenized_ids: torch.Tensor [B, L3] - target tokens
            metadata_list: List of metadata dicts
        """
        waveform_list = []
        instructions = []
        targets = []
        metadata_list = []
        
        for waveform, metadata in batch:
            waveform_list.append(waveform)
            
            # Extract metadata based on stage
            if self.stage == 1:
                # Stage 1: ASR only
                sid, transcript = metadata
                row_dict = {'Transcript': transcript, 'text': transcript}
                instruction, target = get_instruction_and_target(row_dict, "asr")
                metadata_list.append(row_dict)
            elif self.stage == 2:
                # Stage 2: Multi-task - prepare all task data
                sid, gender, age, speed, pitch, energy, emotion, category, transcript, des, ins = metadata
                row_dict = {
                    'Transcript': transcript,
                    'text': transcript,
                    'Des': des,
                    'Ins': ins,
                    'Emotion': emotion
                }
                # Default to ASR for main forward, metadata contains all info
                instruction, target = get_instruction_and_target(row_dict, "asr")
                metadata_list.append(row_dict)
            
            instructions.append(instruction)
            targets.append(target)
        
        # Pad waveform to same length
        # Waveform is 1D [T] - pad time dimension
        max_time_len = max(f.shape[0] for f in waveform_list)
        padded_features = []
        for f in waveform_list:
            if f.shape[0] < max_time_len:
                padding = torch.zeros(max_time_len - f.shape[0])
                f = torch.cat([f, padding], dim=0)  # [max_T]
            padded_features.append(f)
        waveform = torch.stack(padded_features)  # [B, T]
        
        # Tokenize instructions and targets
        pre_tokenized = self.tokenizer(instructions, return_tensors="pt", padding=True, truncation=True)
        output_tokenized = self.tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
        
        # Empty post tokens (can be used for separators if needed)
        batch_size = len(batch)
        post_tokenized_ids = torch.zeros((batch_size, 0), dtype=torch.long)
        
        return (
            waveform,
            pre_tokenized['input_ids'],
            post_tokenized_ids,
            output_tokenized['input_ids'],
            metadata_list
        )


if __name__ == "__main__":
    # Test with dev dataset
    dataset = LibriSpeechDataset(
        jsonl_file='dataset/Librispeech/librispeech_dev.jsonl',
        stage=1
    )
    collator = LibriSpeechCollator(model_name="meta-llama/Llama-3.2-3B-Instruct", stage=1)
    import torch.utils.data as data_utils
    dataloader = data_utils.DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collator, 
        num_workers=4
    )
    for batch in dataloader:
        waveform, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, metadata_list = batch
        print(f"Waveform shape: {waveform.shape}")
        print(f"Pre tokenized shape: {pre_tokenized_ids.shape}")
        print(f"Post tokenized shape: {post_tokenized_ids.shape}")
        print(f"Output tokenized shape: {output_tokenized_ids.shape}")
        print(f"Metadata sample: {metadata_list[0]}")
        break
