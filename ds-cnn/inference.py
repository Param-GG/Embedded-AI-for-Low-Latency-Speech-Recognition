import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import time

def load_models():
    """Load both original and quantized DS-CNN models."""
    try:
        # Load original model
        original_model = tf.keras.models.load_model("ds_cnn_model.h5")
        print("Original model loaded successfully")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="arduino_model_ds_cnn_pruned.tflite")
        interpreter.allocate_tensors()
        print("Quantized model loaded successfully")
        
        return original_model, interpreter
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

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

def run_tflite_inference(interpreter, input_data):
    """Run inference using TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    # Quantize input
    input_data_quantized = np.round(input_data / input_scale + input_zero_point).astype(np.int8)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data_quantized)
    
    # Run inference
    interpreter.invoke()
    
    # Get output and dequantize
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return output_data

def display_predictions(predictions, class_names, model_name, processing_time):
    """Display prediction results in a formatted way."""
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    
    print(f"\n{model_name} Results:")
    if class_names:
        print(f"Predicted Class: {class_names[predicted_label]}")
    print(f"Label Index: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Processing Time: {processing_time:.2f} ms")
    
    # Display top 3 predictions
    if class_names:
        top_indices = np.argsort(predictions)[-3:][::-1]
        print("\nTop 3 Predictions:")
        for idx in top_indices:
            print(f"{class_names[idx]}: {predictions[idx]:.2%}")

def main():
    print("Starting DS-CNN live inference...")
    
    # Load models
    original_model, tflite_interpreter = load_models()
    if not all([original_model, tflite_interpreter]):
        print("Error: Could not load all models. Exiting...")
        return
    
    # Load class names if available
    try:
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_names)} class names")
    except:
        class_names = None
        print("Class names file not found. Will display numeric labels.")
    
    # Print audio configuration
    print(f"\nAudio Configuration:")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"MFCCs: {NUM_MFCCS}")
    print(f"FFT Window Size: {N_FFT}")
    print(f"Hop Length: {HOP_LENGTH}")
    
    inference_times_original = []
    inference_times_quantized = []
    
    while True:
        input("\nPress Enter to record audio...")
        
        try:
            # Record and process audio
            start_time = time.time()
            audio = record_audio(DURATION, SAMPLE_RATE)
            mfcc = extract_mfcc(audio, SAMPLE_RATE)
            input_data = np.expand_dims(mfcc, axis=0)
            
            # Run inference with original model
            original_time = time.time()
            original_pred = original_model.predict(input_data, verbose=0)
            original_time = (time.time() - original_time) * 1000
            inference_times_original.append(original_time)
            
            # Run inference with quantized model
            quantized_time = time.time()
            quantized_pred = run_tflite_inference(tflite_interpreter, input_data)
            quantized_time = (time.time() - quantized_time) * 1000
            inference_times_quantized.append(quantized_time)
            
            # Display results for both models
            display_predictions(original_pred[0], class_names, "Original DS-CNN", original_time)
            display_predictions(quantized_pred[0], class_names, "Quantized DS-CNN", quantized_time)
            
            # Model agreement analysis
            original_label = np.argmax(original_pred)
            quantized_label = np.argmax(quantized_pred)
            print("\nModel Agreement Analysis:")
            if original_label == quantized_label:
                print("Models agree on prediction")
            else:
                print("Models disagree on prediction - check individual results above")
            
            # Display average inference times
            print(f"\nAverage Inference Times:")
            print(f"Original Model: {np.mean(inference_times_original):.2f} ms ± {np.std(inference_times_original):.2f} ms")
            print(f"Quantized Model: {np.mean(inference_times_quantized):.2f} ms ± {np.std(inference_times_quantized):.2f} ms")
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
        
        # Ask to continue
        continue_inference = input("\nDo you want to continue? (y/n): ")
        if continue_inference.lower() != 'y':
            break
    
    # Final statistics
    print("\nFinal Statistics:")
    print(f"Total inferences run: {len(inference_times_original)}")
    print("\nOriginal Model Performance:")
    print(f"Average inference time: {np.mean(inference_times_original):.2f} ms")
    print(f"Standard deviation: {np.std(inference_times_original):.2f} ms")
    print(f"Min time: {np.min(inference_times_original):.2f} ms")
    print(f"Max time: {np.max(inference_times_original):.2f} ms")
    
    print("\nQuantized Model Performance:")
    print(f"Average inference time: {np.mean(inference_times_quantized):.2f} ms")
    print(f"Standard deviation: {np.std(inference_times_quantized):.2f} ms")
    print(f"Min time: {np.min(inference_times_quantized):.2f} ms")
    print(f"Max time: {np.max(inference_times_quantized):.2f} ms")
    
    print("\nExiting DS-CNN live inference...")

if __name__ == "__main__":
    main()