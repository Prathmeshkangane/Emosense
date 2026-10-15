"""
Command-line emotion prediction.

Usage:
    python main.py path/to/audio.wav
    python main.py                      # uses recordings/test_audio.wav
"""
import sys
import json
import numpy as np
import tensorflow as tf
from utils.feature_extraction import extract_features

MODEL_PATH = "models/emotion_model.h5"
SCALER_PATH = "models/scaler.npz"
LABELS_PATH = "models/label_mapping.json"


def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    sc = np.load(SCALER_PATH)
    with open(LABELS_PATH) as f:
        idx_to_emotion = {int(k): v for k, v in json.load(f)["idx_to_emotion"].items()}
    return model, sc["mean"], sc["scale"], idx_to_emotion


def predict_emotion(audio_path, model, mean, scale, idx_to_emotion):
    features = extract_features(audio_path)
    if features is None:
        print("Could not process the audio file (corrupted or too short).")
        return None

    features = (features - mean) / scale                   # same scaling as training
    x = features.reshape(1, features.shape[0], 1)          # (1, 180, 1)
    probs = model.predict(x, verbose=0)[0]

    order = np.argsort(probs)[::-1]
    print(f"Predicted Emotion: {idx_to_emotion[int(order[0])]} ({probs[order[0]] * 100:.1f}%)")
    print("Top 3:", ", ".join(f"{idx_to_emotion[int(i)]} {probs[i] * 100:.1f}%" for i in order[:3]))
    return idx_to_emotion[int(order[0])]


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "recordings/test_audio.wav"
    model, mean, scale, labels = load_assets()
    predict_emotion(audio, model, mean, scale, labels)