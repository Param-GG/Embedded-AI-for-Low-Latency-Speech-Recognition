# live_inference_ds_cnn.py
import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import time

def record_audio(duration=1.0, sample_rate=16000):
    """Record audio from the microphone."""
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc(audio, sample_rate=16000, n_mfcc=12, n_fft=512, hop_length=160):
    """Extract MFCC features and adjust to match the model's input shape."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T  # Shape: (time, n_mfcc)
    # Pad or truncate to 99 frames (as used during training)
    if mfcc.shape[0] < 99:
        pad_width = 99 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:99, :]
    return mfcc.reshape((99, n_mfcc, 1))

def main():
    # Load the DS-CNN model (non-quantized)
    model = tf.keras.models.load_model("ds_cnn_model.h5")
    sample_rate = 16000
    duration = 1.0

    print("Live Inference Mode: Press Ctrl+C to exit.")
    try:
        while True:
            input("Press Enter to record audio...")
            audio = record_audio(duration, sample_rate)
            mfcc = extract_mfcc(audio, sample_rate)
            # Expand dimensions to match model's expected batch shape: (1, 99, n_mfcc, 1)
            input_data = np.expand_dims(mfcc, axis=0)
            predictions = model.predict(input_data)
            predicted_label = np.argmax(predictions)
            confidence = np.max(predictions)
            print(f"Predicted Label: {predicted_label} (Confidence: {confidence:.2f})\n")
    except KeyboardInterrupt:
        print("Exiting live inference mode.")

if __name__ == "__main__":
    main()
