import os
import numpy as np
import librosa
from tqdm import tqdm

# Paths
HOME = os.getcwd()
INPUT_ROOT = f"{HOME}/processed" 
OUTPUT_ROOT = f"{HOME}/logspectrograms"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Audio params
SAMPLE_RATE = 22050
N_FFT = 2048           
HOP_LENGTH = 512       
N_MELS = 128           
FMIN = 20             
FMAX = 8000         


def audio_to_logmel(path):
    """
    Convert audio to log-mel spectrogram
    """
    try:
        # Load audio
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        
        if len(y) == 0:
            raise ValueError("Empty audio")

        # Mel spectrogram (power=2.0 for energy spectrogram)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            power=2.0
        )

        # Convert to log scale (dB)
        # Use ref=1.0 for consistent scale across clips
        logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        
        return logmel
        
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return None


def process_all():
    genres = sorted([g for g in os.listdir(INPUT_ROOT) 
                     if os.path.isdir(os.path.join(INPUT_ROOT, g))])

    if len(genres) == 0:
        print(f"No genre folders found in {INPUT_ROOT}")
        return

    print(f"Converting WAV → Log-Mel Spectrograms...")
    print(f"Params: n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}\n")

    total_spectrograms = 0
    failed = 0

    for genre in tqdm(genres, desc="Genres"):
        genre_in_dir = os.path.join(INPUT_ROOT, genre)
        genre_out_dir = os.path.join(OUTPUT_ROOT, genre)
        os.makedirs(genre_out_dir, exist_ok=True)

        wavs = [f for f in os.listdir(genre_in_dir) if f.endswith(".wav")]
        
        if len(wavs) == 0:
            print(f"No .wav files in {genre}")
            continue

        genre_count = 0
        for wav in tqdm(wavs, desc=f"{genre:15s}", leave=False):
            in_path = os.path.join(genre_in_dir, wav)
            out_path = os.path.join(genre_out_dir, wav.replace(".wav", ".npy"))

            # Skip if already exists
            if os.path.exists(out_path):
                genre_count += 1
                continue

            logmel = audio_to_logmel(in_path)
            
            if logmel is not None:
                # Verify shape
                if logmel.shape[0] != N_MELS:
                    print(f"⚠️  Unexpected shape {logmel.shape} for {wav}")
                    failed += 1
                    continue
                
                np.save(out_path, logmel)
                genre_count += 1
            else:
                failed += 1
        
        total_spectrograms += genre_count
        print(f"{genre}: {genre_count} spectrograms")

    print(f"\nConversion complete!")
    print(f"Total spectrograms: {total_spectrograms}")
    if failed > 0:
        print(f"Failed: {failed}")


def verify_output():
    """Verify the output spectrograms"""
    print("\nVerifying output...")
    
    genres = [g for g in os.listdir(OUTPUT_ROOT) 
              if os.path.isdir(os.path.join(OUTPUT_ROOT, g))]
    
    if len(genres) == 0:
        print("No output folders found!")
        return
    
    total_files = 0
    shapes = []
    values_range = []
    
    for genre in genres:
        genre_dir = os.path.join(OUTPUT_ROOT, genre)
        npy_files = [f for f in os.listdir(genre_dir) if f.endswith('.npy')]
        total_files += len(npy_files)
        
        # Sample a few files to check
        for npy_file in npy_files[:3]:
            arr = np.load(os.path.join(genre_dir, npy_file))
            shapes.append(arr.shape)
            values_range.append((arr.min(), arr.max()))
    
    print(f"Total .npy files: {total_files}")
    print(f"Sample shapes: {set(shapes)}")
    print(f"Value ranges (min, max): {values_range[:3]}")
    
    # Check consistency
    if len(set(shapes)) > 1:
        print(" WARNING: Inconsistent shapes detected!")
    else:
        print(f"All spectrograms have shape: {shapes[0]}")


if __name__ == "__main__":
    process_all()
    verify_output()