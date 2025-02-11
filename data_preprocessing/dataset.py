# dataset.py
import tensorflow as tf
import numpy as np
from pathlib import Path
from audio_processing import preprocess_audio
from utils import load_config

# Load configuration to get NUM_MFCCS for padded batching
config = load_config()
NUM_MFCCS = config["num_mfccs"]

def prepare_speech_commands_dataset(data_dir, batch_size=32, validation_split=0.1, test_split=0.1, seed=123):
    """
    Prepare the Speech Commands dataset for training, validation, and testing.
    
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for training and validation.
        validation_split (float): Proportion of the data for validation.
        test_split (float): Proportion of the data for testing.
        seed (int): Random seed for shuffling and splitting.
        
    Returns:
        tuple: (training dataset, validation dataset, test dataset, list of class names).
    """
    data_dir = Path(data_dir)

    # Exclude '_background_noise_' from class names
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name != "_background_noise_"])
    
    file_paths = []
    labels = []

    # Collect all file paths and corresponding labels
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for file_path in list(class_dir.glob("*.wav")) + list(class_dir.glob("*.WAV")):  # Case-insensitive globbing
            file_paths.append(str(file_path))
            labels.append(label)

    # Convert file paths and labels to TensorFlow constants
    file_paths = tf.constant(file_paths, dtype=tf.string)
    labels = tf.constant(labels, dtype=tf.int32)  # Ensure labels are int32

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

    train_file_paths = tf.gather(file_paths, train_indices)
    train_labels = tf.gather(labels, train_indices)

    val_file_paths = tf.gather(file_paths, val_indices)
    val_labels = tf.gather(labels, val_indices)

    test_file_paths = tf.gather(file_paths, test_indices)
    test_labels = tf.gather(labels, test_indices)

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_file_paths, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))

    # Apply preprocessing to extract MFCC features
    train_ds = train_ds.map(lambda x, y: preprocess_audio(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: preprocess_audio(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: preprocess_audio(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    # Add channel dimension to MFCCs
    train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
    val_ds = val_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
    test_ds = test_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))

    # Ensure labels are int32 before batching
    train_ds = train_ds.map(lambda x, y: (x, tf.cast(y, tf.int32)))
    val_ds = val_ds.map(lambda x, y: (x, tf.cast(y, tf.int32)))
    test_ds = test_ds.map(lambda x, y: (x, tf.cast(y, tf.int32)))

    # Apply padded batching
    train_ds = train_ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None, NUM_MFCCS, 1], []),
        padding_values=(0.0, tf.constant(0, dtype=tf.int32))  # Ensure padding value for labels is int32
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None, NUM_MFCCS, 1], []),
        padding_values=(0.0, tf.constant(0, dtype=tf.int32))
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None, NUM_MFCCS, 1], []),
        padding_values=(0.0, tf.constant(0, dtype=tf.int32))
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
