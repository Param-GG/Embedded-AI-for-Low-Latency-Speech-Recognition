# dataset.py
import tensorflow as tf
import numpy as np
from pathlib import Path
from data_preprocessing.audio_processing import preprocess_audio
from data_preprocessing.utils import load_config
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.audio_processing import preprocess_audio


# Load configuration to get NUM_MFCCS for padded batching
config = load_config()
NUM_MFCCS = config["num_mfccs"]

def prepare_speech_commands_dataset(data_dir, batch_size=32, validation_split=0.1, test_split=0.1, seed=123):
    """
    Prepare the Speech Commands dataset with robust error handling.
    """
    data_dir = Path(data_dir)

    # Exclude '_background_noise_' from class names
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name != "_background_noise_"])
    
    file_paths = []
    labels = []
    invalid_files = []

    # Collect all file paths and corresponding labels with validation
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        wav_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.WAV"))
        
        for file_path in wav_files:
            try:
                # Quick validation of file
                audio_binary = tf.io.read_file(str(file_path))
                tf.audio.decode_wav(audio_binary)
                file_paths.append(str(file_path))
                labels.append(label)
            except Exception as e:
                invalid_files.append((str(file_path), str(e)))
                continue

    if invalid_files:
        print(f"\nFound {len(invalid_files)} invalid files:")
        for file, error in invalid_files:
            print(f" - {file}")
            
    if not file_paths:
        raise ValueError("No valid audio files found in the dataset")

    # Convert to TensorFlow constants
    file_paths = tf.constant(file_paths, dtype=tf.string)
    labels = tf.constant(labels, dtype=tf.int32)

    # Shuffle and split dataset
    dataset_size = len(file_paths)
    val_size = int(dataset_size * validation_split)
    test_size = int(dataset_size * test_split)

    indices = np.arange(dataset_size)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[val_size + test_size:]
    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size + test_size]

    # Create split datasets
    def create_split_dataset(indices):
        split_paths = tf.gather(file_paths, indices)
        split_labels = tf.gather(labels, indices)
        ds = tf.data.Dataset.from_tensor_slices((split_paths, split_labels))
        
        # Apply preprocessing with error handling
        ds = ds.map(
            lambda x, y: preprocess_audio(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Add channel dimension
        ds = ds.map(
            lambda x, y: (tf.expand_dims(x, -1), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Ensure labels are int32
        ds = ds.map(
            lambda x, y: (x, tf.cast(y, tf.int32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply padded batching
        ds = ds.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None, NUM_MFCCS, 1], []),
            padding_values=(0.0, tf.constant(0, dtype=tf.int32))
        ).prefetch(tf.data.AUTOTUNE)
        
        return ds

    # Create final datasets
    train_ds = create_split_dataset(train_indices)
    val_ds = create_split_dataset(val_indices)
    test_ds = create_split_dataset(test_indices)

    print(f"\nDataset splits:")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Number of classes: {len(class_names)}")

    return train_ds, val_ds, test_ds, class_names