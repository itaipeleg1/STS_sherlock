import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import torch
import librosa

os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'

# Import transformers for WavLM model
from transformers import AutoModelForAudioClassification

print("Loading WavLM Multi-Attributes emotion model...")
model = AutoModelForAudioClassification.from_pretrained(
    "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
    trust_remote_code=True
).to("cuda:0")
model.eval()

print(f"Model sampling rate: {model.config.sampling_rate}")
print(f"Model normalization - mean: {model.config.mean:.4f}, std: {model.config.std:.4f}")
print("Model outputs: 0=arousal, 1=dominance, 2=valence")

# Read the audio file
audio, sr = sf.read("/home/new_storage/sherlock/data/sound_track.wav")
print(f"\nAudio length: {len(audio)} samples")
print(f"Original sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")

# Convert stereo to mono by averaging channels
if audio.ndim > 1:
    audio_mono = np.mean(audio, axis=1)
else:
    audio_mono = audio

# Resample to model's sample rate (16kHz)
target_sr = model.config.sampling_rate
if sr != target_sr:
    print(f"Resampling from {sr} Hz to {target_sr} Hz...")
    audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=target_sr)
    sr = target_sr
    print(f"New audio length: {len(audio_mono)} samples")

# Calculate chunk size for 3 seconds
chunk_duration = 3.0  # seconds
chunk_size = int(sr * chunk_duration)  # 48000 samples at 16kHz
print(f"\nChunk size: {chunk_size} samples ({chunk_duration} seconds)")

output_dir = "/home/new_storage/sherlock/STS_sherlock/projects data/annotations"
os.makedirs(output_dir, exist_ok=True)

arousal_scores = []
dominance_scores = []
valence_scores = []

print("\nProcessing audio chunks...")
with torch.no_grad():
    for i in tqdm(range(0, len(audio_mono) - chunk_size, chunk_size)):
        chunk = audio_mono[i:i+chunk_size]

        # Normalize using model's mean/std
        mean = model.config.mean
        std = model.config.std
        norm_chunk = (chunk - mean) / (std + 0.000001)

        # Prepare tensors
        wavs = torch.tensor(norm_chunk, dtype=torch.float32).unsqueeze(0).to("cuda:0")
        mask = torch.ones(1, len(norm_chunk)).to("cuda:0")

        # Get predictions: [arousal, dominance, valence]
        pred = model(wavs, mask)
        pred_array = pred[0].cpu().numpy()  # Shape: (3,)

        arousal_scores.append(pred_array[0])
        dominance_scores.append(pred_array[1])
        valence_scores.append(pred_array[2])

print(f"\nProcessed {len(arousal_scores)} emotion predictions")

# Convert to numpy arrays
arousal_array = np.array(arousal_scores)
dominance_array = np.array(dominance_scores)
valence_array = np.array(valence_scores)

print(f"\nArousal range: [{arousal_array.min():.3f}, {arousal_array.max():.3f}], mean: {arousal_array.mean():.3f}")
print(f"Dominance range: [{dominance_array.min():.3f}, {dominance_array.max():.3f}], mean: {dominance_array.mean():.3f}")
print(f"Valence range: [{valence_array.min():.3f}, {valence_array.max():.3f}], mean: {valence_array.mean():.3f}")

# Repeat each score twice to match 1.5s fMRI TR
# 3-second chunks → two 1.5-second TRs per chunk
arousal_1_5s = np.repeat(arousal_array, 2).reshape(-1, 1)
dominance_1_5s = np.repeat(dominance_array, 2).reshape(-1, 1)
valence_1_5s = np.repeat(valence_array, 2).reshape(-1, 1)

print(f"\nAfter repeating for 1.5s TR: {arousal_1_5s.shape[0]} time points")

# Save individual dimensions
arousal_path = os.path.join(output_dir, "arousal_scores.npy")
np.save(arousal_path, arousal_1_5s)
print(f"\nSaved arousal scores with shape {arousal_1_5s.shape} to: {arousal_path}")

dominance_path = os.path.join(output_dir, "dominance_scores.npy")
np.save(dominance_path, dominance_1_5s)
print(f"Saved dominance scores with shape {dominance_1_5s.shape} to: {dominance_path}")

valence_path = os.path.join(output_dir, "valence_scores.npy")
np.save(valence_path, valence_1_5s)
print(f"Saved valence scores with shape {valence_1_5s.shape} to: {valence_path}")

# Save all dimensions combined
all_dimensions = np.hstack([arousal_1_5s, dominance_1_5s, valence_1_5s])
all_path = os.path.join(output_dir, "emotion_dimensions_all.npy")
np.save(all_path, all_dimensions)
print(f"Saved all emotion dimensions with shape {all_dimensions.shape} to: {all_path}")

# Create derived measures
emotionality = arousal_1_5s  # Arousal as direct emotionality measure
emotional_positivity = (arousal_1_5s * valence_1_5s)  # High arousal + positive
emotional_negativity = (arousal_1_5s * (1 - valence_1_5s))  # High arousal + negative

emotionality_path = os.path.join(output_dir, "emotionality_scores.npy")
np.save(emotionality_path, emotionality)
print(f"Saved emotionality (arousal) scores to: {emotionality_path}")

pos_path = os.path.join(output_dir, "emotional_positivity_scores.npy")
np.save(pos_path, emotional_positivity)
print(f"Saved emotional positivity (arousal × valence) scores to: {pos_path}")

neg_path = os.path.join(output_dir, "emotional_negativity_scores.npy")
np.save(neg_path, emotional_negativity)
print(f"Saved emotional negativity (arousal × (1-valence)) scores to: {neg_path}")

print("\n✓ Done! All emotion scores saved.")
