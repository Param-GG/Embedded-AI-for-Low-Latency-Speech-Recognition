import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import time

# Load the trained DS-CNN model
model = tf.keras.models.load_model("ds_cnn_model.h5")

# Parameters for recording and feature extraction
DURATION = 1.0         # Duration of audio in seconds
SAMPLE_RATE = 16000    # Sample rate for audio
NUM_MFCCS = 12         # Number of MFCCs to extract
N_FFT = 512            # FFT window size
HOP_LENGTH = 160       # Hop length for MFCC extraction

def record_audio(duration, sample_rate):
    """Record audio from the microphone."""
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is done
    return np.squeeze(audio)

def extract_mfcc(audio, sample_rate, n_mfcc=NUM_MFCCS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Extract MFCC features from audio."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T  # Transpose to shape (time, n_mfcc)
    # Pad or truncate to match input shape expected by the model (99, 12, 1)
    if mfcc.shape[0] < 99:
        pad_width = 99 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:99, :]
    return mfcc.reshape((99, n_mfcc, 1))

def main():
    print("Starting live inference...")
    while True:
        input("Press Enter to record audio...")  # Wait for user input to start recording
        # Record audio
        audio = record_audio(DURATION, SAMPLE_RATE)
        # Extract MFCC features
        mfcc = extract_mfcc(audio, SAMPLE_RATE)
        # Expand dimensions to create a batch of 1: (1, 99, 12, 1)
        input_data = np.expand_dims(mfcc, axis=0)
        # Run inference
        predictions = model.predict(input_data)
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions)
        print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}")
        
        # Optionally, add a condition to break the loop
        continue_inference = input("Do you want to continue? (y/n): ")
        if continue_inference.lower() != 'y':
            break
    print("Exiting live inference...")

if __name__ == "__main__":
    main()
