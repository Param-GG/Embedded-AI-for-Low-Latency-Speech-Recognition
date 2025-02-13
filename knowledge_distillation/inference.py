import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import time

# Parameters for recording and feature extraction
DURATION = 1.0         # Duration of audio in seconds
SAMPLE_RATE = 16000    # Sample rate for audio
NUM_MFCCS = 12         # Number of MFCCs to extract
N_FFT = 512            # FFT window size
HOP_LENGTH = 160       # Hop length for MFCC extraction

def load_models():
    """Load both teacher and student models."""
    try:
        teacher_model = tf.keras.models.load_model("teacher_model.h5")
        student_model = tf.keras.models.load_model("student_model.h5")
        return teacher_model, student_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

def load_tflite_model():
    """Load the quantized TFLite student model."""
    try:
        interpreter = tf.lite.Interpreter(model_path="arduino_model_student_pruned.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {str(e)}")
        return None

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

def main():
    print("Starting knowledge distillation live inference...")
    
    # Load models
    teacher_model, student_model = load_models()
    tflite_interpreter = load_tflite_model()
    
    if not all([teacher_model, student_model, tflite_interpreter]):
        print("Error: Could not load all models. Exiting...")
        return
    
    print("All models loaded successfully.")
    
    # Load class names if available
    try:
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except:
        class_names = None
        print("Class names file not found. Will display numeric labels.")
    
    while True:
        input("\nPress Enter to record audio...")
        
        try:
            # Record and process audio
            start_time = time.time()
            audio = record_audio(DURATION, SAMPLE_RATE)
            mfcc = extract_mfcc(audio, SAMPLE_RATE)
            input_data = np.expand_dims(mfcc, axis=0)
            
            # Run inference on all models
            print("\nRunning inference on all models...")
            
            # Teacher model inference
            teacher_time = time.time()
            teacher_pred = teacher_model.predict(input_data, verbose=0)
            teacher_time = (time.time() - teacher_time) * 1000
            
            # Student model inference
            student_time = time.time()
            student_pred = student_model.predict(input_data, verbose=0)
            student_time = (time.time() - student_time) * 1000
            
            # Quantized student model inference
            tflite_time = time.time()
            tflite_pred = run_tflite_inference(tflite_interpreter, input_data)
            tflite_time = (time.time() - tflite_time) * 1000
            
            # Get predictions and confidences
            teacher_label = np.argmax(teacher_pred)
            student_label = np.argmax(student_pred)
            tflite_label = np.argmax(tflite_pred)
            
            # Display results
            print("\nInference Results:")
            print("\nTeacher Model:")
            if class_names:
                print(f"Predicted Class: {class_names[teacher_label]}")
            print(f"Label Index: {teacher_label}")
            print(f"Confidence: {np.max(teacher_pred):.2%}")
            print(f"Processing Time: {teacher_time:.2f} ms")
            
            print("\nStudent Model:")
            if class_names:
                print(f"Predicted Class: {class_names[student_label]}")
            print(f"Label Index: {student_label}")
            print(f"Confidence: {np.max(student_pred):.2%}")
            print(f"Processing Time: {student_time:.2f} ms")
            
            print("\nQuantized Student Model:")
            if class_names:
                print(f"Predicted Class: {class_names[tflite_label]}")
            print(f"Label Index: {tflite_label}")
            print(f"Confidence: {np.max(tflite_pred):.2%}")
            print(f"Processing Time: {tflite_time:.2f} ms")
            
            # Agreement analysis
            print("\nModel Agreement Analysis:")
            all_agree = (teacher_label == student_label == tflite_label)
            print(f"All models agree: {'Yes' if all_agree else 'No'}")
            if not all_agree:
                print("Disagreement detected. Check individual predictions above.")
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
        
        # Ask to continue
        continue_inference = input("\nDo you want to continue? (y/n): ")
        if continue_inference.lower() != 'y':
            break
            
    print("Exiting knowledge distillation live inference...")

if __name__ == "__main__":
    main()