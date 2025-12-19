import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import os
import torch
from transformers import AutoTokenizer

def preprocess_text(text):
    """Preprocess text for downstream tasks"""
    if pd.isna(text):
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
        row: DataFrame row with all metadata
        task: "asr", "sed", or "ser"
    
    Returns:
        instruction: str, target: str
    """
    if task == "asr":
        return "Transcribe: ", preprocess_text(row['Transcript'])
    elif task == "sed":
        return row.get('Ins', 'Describe the emotional characteristics of this speech: '), preprocess_text(row.get('Des', ''))
    elif task == "ser":
        return "Classify emotion: ", row['Emotion'].lower() if pd.notna(row['Emotion']) else ''
    else:
        raise ValueError(f"Unknown task: {task}")

class SpeechCraftDataset(Dataset):
    def __init__(self, csv_file, stage=1):
        """
        Dataset for SpeechCraft with speed optimization.

        Each row in the CSV should have the following columns:
            sid:         Sample ID
            audio_path:  Path to the audio file
            Gender:      Speaker gender (e.g., male/female)
            Age:         Speaker age or age group
            Speed:       Speaking speed (could be a category or value)
            Pitch:       Pitch characteristics (e.g., high/low)
            Energy:      Overall energy level of the utterance
            Emotion:     Expressed emotion (e.g., happy, sad)
            Category:    Category label of the sample
            Transcript:  Text transcript of the audio
            Des:         Additional description or metadata
            Ins:         Instruction or related prompt

        The 'metadata' returned by __getitem__ is a list containing all of the above in this order:
            [Gender, Age, Speed, Pitch, Energy, Emotion, Category, Transcript, Des, Ins]
        """
        self.stage = stage
        # self.hubert_features_dir = hubert_features_dir
        
        if self.stage == 1:
            self.labels = ['sid', 'Transcript']
        elif self.stage == 2:
            self.labels = ['sid', 'Gender', 'Age', 'Speed', 'Pitch', 'Energy', 'Emotion', 'Category', 'Transcript', 'Des', 'Ins']
        self.audio_metadata = pd.read_csv(csv_file)

    def _load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
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
        return len(self.audio_metadata)
    
    def __getitem__(self, idx):
        row = self.audio_metadata.iloc[idx]
        audio_path = row['audio_path']
        audio = self._load_audio(audio_path)
        metadata = tuple(row[label] for label in self.labels)
        return audio, metadata



class MyCollator:
    """Collator for batching HuBERT features with instructions and targets"""
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
                row_dict = {'Transcript': transcript}
                instruction, target = get_instruction_and_target(row_dict, "asr")
                metadata_list.append(row_dict)
            elif self.stage == 2:
                # Stage 2: Multi-task - prepare all task data
                sid, gender, age, speed, pitch, energy, emotion, category, transcript, des, ins = metadata
                row_dict = {
                    'Transcript': transcript,
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
    dataset = SpeechCraftDataset(csv_file='dataset/SpeechCraft_final_df.csv', stage=1)
    my_collator = MyCollator(model_name="meta-llama/Meta-Llama-3-8B", stage=1)
    import torch.utils.data as data_utils
    dataloader = data_utils.DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=my_collator, 
        num_workers=4
    )
    for batch in dataloader:
        waveform, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, metadata_list = batch
        print(waveform.shape)
        print(pre_tokenized_ids.shape)
        print(post_tokenized_ids.shape)
        print(output_tokenized_ids.shape)
        print(metadata_list)
        break