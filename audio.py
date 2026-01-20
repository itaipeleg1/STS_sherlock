import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import os
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor
import torch
import librosa  # Add this for resampling

os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'

# Read the audio file
audio, sr = sf.read("/home/new_storage/sherlock/data/sound_track.wav")
print(f"Audio length: {len(audio)} samples")
print(f"Original sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")

# Convert stereo to mono by averaging channels
if audio.ndim > 1:
    audio_mono = np.mean(audio, axis=1)
else:
    audio_mono = audio

# Resample to 48kHz (CLAP's required sample rate)
target_sr = 48000
if sr != target_sr:
    print(f"Resampling from {sr} Hz to {target_sr} Hz...")
    audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=target_sr)
    sr = target_sr
    print(f"New audio length: {len(audio_mono)} samples")

model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(0)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

# Calculate chunk size for 1.5 seconds at 48kHz
chunk_duration = 1.5  # seconds
chunk_size = int(sr * chunk_duration)  # 72000 samples at 48kHz
print(f"Chunk size: {chunk_size} samples ({chunk_duration} seconds)")

output_dir = "/home/new_storage/sherlock/project_data/audio_embeddings"
os.makedirs(output_dir, exist_ok=True)

embeddings_list = []

with torch.no_grad():
    for i in tqdm(range(0, len(audio_mono) - chunk_size, chunk_size)):
        chunk = audio_mono[i:i+chunk_size]
        
        # Process and get embeddings (now at 48kHz)
        inputs = processor(audios=chunk, return_tensors="pt", sampling_rate=sr).to(0)
        audio_embed = model.get_audio_features(**inputs)
        
        # Squeeze to make it 1D: shape (1, 512) -> (512,)
        audio_embed_1d = audio_embed.squeeze().cpu().numpy()
        
        # Save individual embeddings
        np.save(os.path.join(output_dir, f"audio_embed_{i//chunk_size:04d}.npy"), audio_embed_1d)


print(f"\nSaved {len(embeddings_list)} embeddings")
print(f"Each embedding shape: {embeddings_list[0].shape}")
