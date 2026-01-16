import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

HOME = os.getcwd()

INPUT_ROOT = f"{HOME}/genres_original"
OUTPUT_ROOT = f"{HOME}/processed"
TARGET_SR = 22050
CLIP_DURATION = 3  # seconds
SAMPLES_PER_CLIP = TARGET_SR * CLIP_DURATION

def rms_normalize(audio, target_rms=0.05):
    """
    RMS normalization to consistent loudness
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio
    normalized = audio * (target_rms / rms)
    # Clip to prevent values exceeding [-1, 1]
    return np.clip(normalized, -1.0, 1.0)


def process_file(in_path, out_dir):
    try:
        audio, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)
        
        if len(audio) == 0:
            print(f"Empty audio file: {in_path}")
            return

        # Normalize loudness BEFORE chunking
        audio = rms_normalize(audio, target_rms=0.05)

        # Pad end so final chunk fits fully
        if len(audio) % SAMPLES_PER_CLIP != 0:
            pad = SAMPLES_PER_CLIP - (len(audio) % SAMPLES_PER_CLIP)
            audio = np.pad(audio, (0, pad), mode='constant')

        # Chunk the audio
        num_chunks = len(audio) // SAMPLES_PER_CLIP
        base = os.path.splitext(os.path.basename(in_path))[0]

        for i in range(num_chunks):
            chunk = audio[i * SAMPLES_PER_CLIP : (i + 1) * SAMPLES_PER_CLIP]
            
            # Skip silent chunks (RMS < threshold)
            chunk_rms = np.sqrt(np.mean(chunk**2))
            if chunk_rms < 1e-4:
                continue
            
            out_path = os.path.join(out_dir, f"{base}_chunk{i:03d}.wav")
            sf.write(out_path, chunk, TARGET_SR)
            
    except Exception as e:
        print(f"Failed to process {in_path}: {e}")
        return


def preprocess_dataset():
    genres = [g for g in os.listdir(INPUT_ROOT) 
              if os.path.isdir(os.path.join(INPUT_ROOT, g))]

    if len(genres) == 0:
        print(f"No genre folders found in {INPUT_ROOT}")
        return

    print(f"📂 Found {len(genres)} genres: {genres}\n")
    print("Starting audio preprocessing...\n")

    total_chunks = 0

    # Process each genre
    for genre in tqdm(genres, desc="Genres", unit="genre"):
        genre_path = os.path.join(INPUT_ROOT, genre)
        out_genre_dir = os.path.join(OUTPUT_ROOT, genre)
        os.makedirs(out_genre_dir, exist_ok=True)

        # Collect wav files
        wav_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        
        if len(wav_files) == 0:
            print(f"⚠️  No .wav files in {genre}")
            continue

        # Process files in genre
        genre_chunks = 0
        for file in tqdm(wav_files, desc=f"{genre:15s}", unit="file", leave=False):
            in_path = os.path.join(genre_path, file)
            before = len([f for f in os.listdir(out_genre_dir) if f.endswith('.wav')])
            process_file(in_path, out_genre_dir)
            after = len([f for f in os.listdir(out_genre_dir) if f.endswith('.wav')])
            genre_chunks += (after - before)
        
        total_chunks += genre_chunks
        print(f"{genre}: {genre_chunks} chunks created")

    print(f"\nTotal chunks created: {total_chunks}")

if __name__ == "__main__":
    preprocess_dataset()
