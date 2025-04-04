import sounddevice as sd
import librosa
import numpy as np
import joblib

# Load the trained model
rf_model = joblib.load("random_forest_slap_classifier.pkl")

# Audio settings
SAMPLE_RATE = 44100  # Standard audio sample rate
FRAME_SIZE = 2048  # Window size for feature extraction
HOP_SIZE = 512  # Hop length for overlapping windows

def extract_mfcc(audio_segment, sr):
    """Extract MFCCs from an audio segment."""
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)  # Take mean across time

def predict_slapping(audio_segment):
    """Predict whether the segment contains a slap or pop."""
    mfcc_features = extract_mfcc(audio_segment, SAMPLE_RATE)
    mfcc_features = mfcc_features.reshape(1, -1)  # Reshape for prediction
    predicted_label = rf_model.predict(mfcc_features)[0]
    return predicted_label

def callback(indata, frames, time, status):
    """Callback function that runs every audio buffer."""
    if status:
        print(status)
    
    # Convert audio to mono if it's stereo
    audio_segment = indata[:, 0] if indata.ndim > 1 else indata

    # Predict slap technique
    technique = predict_slapping(audio_segment)
    
    # Print detected technique
    print(f"🎸 Detected Slap Technique: {technique}")

# Start streaming audio from the microphone
print("🎧 Listening for slap bass... (Press Ctrl+C to stop)")
with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
    while True:
        pass  # Keep running
