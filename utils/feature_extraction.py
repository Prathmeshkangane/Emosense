"""
Feature extraction for Speech Emotion Recognition.

IMPORTANT: this file and the feature code inside train_colab.py must stay
IDENTICAL (same SR, DURATION, OFFSET, N_MFCC and same steps). If they differ,
the model sees different numbers at prediction time than it saw in training.

Output: a (180,) float32 vector = MFCC(40) + Chroma(12) + log-Mel(128),
each averaged over time.
"""
import numpy as np
import librosa

SR = 22050
DURATION = 3
OFFSET = 0.5
N_MFCC = 40
FEATURE_DIM = 180


def features_from_signal(y):
    """y: 1-D float waveform sampled at SR. Returns a (180,) float32 array."""
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC), axis=1)      # (40,)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=SR), axis=1)            # (12,)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR))        # log scale
    mel = np.mean(mel, axis=1)                                                   # (128,)
    feats = np.hstack((mfcc, chroma, mel)).astype(np.float32)
    return np.nan_to_num(feats)


def extract_features(audio_path):
    """Load an audio file and return its (180,) feature vector, or None on failure."""
    try:
        y, _ = librosa.load(audio_path, sr=SR, duration=DURATION, offset=OFFSET)
        if len(y) < SR * 0.5:          # clip too short after the offset
            return None
        return features_from_signal(y)
    except Exception as e:
        print(f"[feature_extraction] Error: {e}")
        return None