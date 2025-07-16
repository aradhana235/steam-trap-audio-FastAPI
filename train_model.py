import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# ✅ FIXED Initialization
X = []
y = []

base_dir = "audio_samples"

for label in os.listdir(base_dir):
    folder = os.path.join(base_dir, label)
    print(f"Checking folder: {folder}")
    for f in os.listdir(folder):
        if f.endswith(".mp3") or f.endswith(".wav"):
            file_path = os.path.join(folder, f)
            print(f"  Loading: {file_path}")
            try:
                features = extract_features(file_path)
                print(f"  ✅ Features shape: {features.shape}")
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"  ❌ Error with {f}: {e}")

print(f"\nTotal files processed: {len(X)}")
print(f"Labels: {set(y)}")

if len(X) > 0:
    clf = RandomForestClassifier()
    clf.fit(X, y)

    os.makedirs("model", exist_ok=True)  # ensure folder exists
    joblib.dump(clf, "model/audio_model.pkl")
    print("✅ Model trained and saved.")
else:
    print("⚠️ No audio data found. Please check your 'audio_samples' folder.")
