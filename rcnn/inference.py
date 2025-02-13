import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import time

# Load the trained RCNN model (either original or pruned)
model = tf.keras.models.load_model("rcnn_model.h5")

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
    print("Starting RCNN live inference...")
    print("Model loaded and ready for inference.")
    
    # Load class names if available
    try:
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except:
        class_names = None
        print("Class names file not found. Will display numeric labels.")
    
    while True:
        input("\nPress Enter to record audio...")  # Wait for user input to start recording
        
        try:
            # Record audio
            start_time = time.time()
            audio = record_audio(DURATION, SAMPLE_RATE)
            
            # Extract MFCC features
            mfcc = extract_mfcc(audio, SAMPLE_RATE)
            
            # Expand dimensions to create a batch of 1: (1, 99, 12, 1)
            input_data = np.expand_dims(mfcc, axis=0)
            
            # Run inference
            predictions = model.predict(input_data, verbose=0)
            predicted_label = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Display results
            print("\nInference Results:")
            if class_names:
                print(f"Predicted Class: {class_names[predicted_label]}")
            print(f"Label Index: {predicted_label}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Processing Time: {processing_time:.2f} ms")
            
            # Display top 3 predictions if class names are available
            if class_names:
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                print("\nTop 3 Predictions:")
                for idx in top_indices:
                    print(f"{class_names[idx]}: {predictions[0][idx]:.2%}")
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
        
        # Ask to continue
        continue_inference = input("\nDo you want to continue? (y/n): ")
        if continue_inference.lower() != 'y':
            break
            
    print("Exiting RCNN live inference...")

if __name__ == "__main__":
    main()