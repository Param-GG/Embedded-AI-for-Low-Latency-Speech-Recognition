import tensorflow as tf
import numpy as np
from data_preprocessing.audio_processing import preprocess_audio
from pathlib import Path

def predict_audio(model, audio_file, class_names):
    """
    Make prediction on a single audio file.
    """
    try:
        # Preprocess audio file
        audio_binary = tf.io.read_file(audio_file)
        waveform, _ = tf.audio.decode_wav(audio_binary)
        mfccs = preprocess_audio(waveform)
        mfccs = tf.expand_dims(mfccs, 0)  # Add batch dimension
        mfccs = tf.expand_dims(mfccs, -1)  # Add channel dimension

        # Make prediction
        predictions = model.predict(mfccs, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_word = class_names[predicted_index]
        confidence = predictions[0][predicted_index]

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            (class_names[i], float(predictions[0][i])) 
            for i in top_3_indices
        ]

        return predicted_word, confidence, top_3_predictions
    
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        return None, None, None

def main():
    # Load model
    model = tf.keras.models.load_model("ds_cnn_model.h5")
    
    # Get class names from your dataset directory
    data_dir = Path('datasets/speech_commands_v0_extracted')
    class_names = sorted([d.name for d in data_dir.iterdir() 
                         if d.is_dir() and d.name != "_background_noise_"])
    
    # Test files - update these paths to your test audio files
    test_files = [
        "path/to/test/audio1.wav",
        "path/to/test/audio2.wav"
    ]
    
    # Run inference
    for audio_file in test_files:
        word, conf, top_3 = predict_audio(model, audio_file, class_names)
        if word is not None:
            print(f"\nAudio file: {audio_file}")
            print(f"Predicted word: {word} (confidence: {conf:.2%})")
            print("Top 3 predictions:")
            for word, conf in top_3:
                print(f"  {word}: {conf:.2%}")

if __name__ == "__main__":
    main()