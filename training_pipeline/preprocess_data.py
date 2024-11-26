import os
import tensorflow as tf
import numpy as np
import random
from pathlib import Path

# Signal and feature extraction parameters
SAMPLE_RATE = 8000  # 8 kHz
FRAME_SIZE = 0.032  # 32 ms
FRAME_STRIDE = 0.02  # 20 ms
NFFT = 256  # FFT size, equal to frame length
NUM_MEL_BINS = 40  # Assuming unchanged; the paper does not specify this
NUM_MFCCS = 12  # Number of MFCC coefficients remains standard
BACKGROUND_NOISE_DIR = "./datasets/speech_commands_v0.02/_background_noise_"


def add_background_noise(audio):
    """
    Adds background noise to the audio signal.
    Args:
        audio (tf.Tensor): Input audio signal (1D tensor).
        sample_rate (int): Sampling rate of the audio.
    Returns:
        tf.Tensor: Augmented audio with background noise.
    """
    # Get all background noise files
    noise_files = list(Path(BACKGROUND_NOISE_DIR).glob("*.wav"))
    if not noise_files:
        return (
            audio  # If no background noise files are available, return original audio
        )

    # Randomly select a background noise file
    selected_noise_file = random.choice(noise_files)

    # Load the background noise file
    noise_audio_binary = tf.io.read_file(str(selected_noise_file))
    noise_waveform, _ = tf.audio.decode_wav(noise_audio_binary, desired_channels=1)
    noise_waveform = tf.squeeze(noise_waveform, axis=-1)  # Remove channel dimension
    noise_waveform = tf.cast(noise_waveform, tf.float32)

    # Ensure the background noise is at least as long as the target audio
    noise_waveform = tf.tile(
        noise_waveform, [tf.math.ceil(tf.shape(audio)[0] / tf.shape(noise_waveform)[0])]
    )
    noise_waveform = noise_waveform[
        : tf.shape(audio)[0]
    ]  # Trim to match target audio length

    # Scale background noise to 10% of the target audio's RMS volume
    audio_rms = tf.math.sqrt(tf.reduce_mean(tf.square(audio)))
    noise_rms = tf.math.sqrt(tf.reduce_mean(tf.square(noise_waveform)))
    noise_waveform = noise_waveform * (0.1 * audio_rms / noise_rms)

    # Add the scaled background noise to the target audio
    augmented_audio = audio + noise_waveform

    return augmented_audio


# * experimenting
import matplotlib.pyplot as plt


def visualize_waveform(waveform):
    """
    Visualize the waveform after ensuring it's in numpy format.
    """
    if tf.executing_eagerly():
        # Convert tensor to numpy if in eager execution mode
        waveform = waveform.numpy() if isinstance(waveform, tf.Tensor) else waveform
    else:
        # For graph execution, use TensorFlow operations
        waveform = tf.make_ndarray(tf.make_tensor_proto(waveform))

    # Proceed with visualization using matplotlib or similar
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(waveform)
    plt.title("Waveform Visualization")
    plt.show()


def visualize_spectrogram(spectrogram):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Power")
    plt.title("Power Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.show()


def visualize_log_mel_spectrogram(log_mel_spectrogram):
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel_spectrogram.T, aspect="auto", origin="lower", cmap="inferno")
    plt.colorbar(label="Log Amplitude")
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.show()


def visualize_mfccs(mfccs):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs.T, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCCs")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.show()


# * experimenting


def preprocess_audio(file_path, label):
    """
    Preprocess an audio file to extract MFCC features.
    Args:
        file_path (str): Path to the audio file.
        label (int): Integer label for the audio file's class.
    Returns:
        tuple: (MFCCs as a TensorFlow tensor, label as an integer).
    """
    # Step 1: Read audio file
    audio_binary = tf.io.read_file(file_path)
    waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)  # Remove channel dimension
    waveform = tf.cast(waveform, tf.float32)

    # Step 2: Add background noise
    waveform = add_background_noise(waveform)

    # Step 3: Pre-emphasis
    pre_emphasis = 0.9375
    waveform = tf.concat(
        [[waveform[0]], waveform[1:] - pre_emphasis * waveform[:-1]], axis=0
    )

    # * experimenting
    # visualize_waveform(waveform)

    # Step 4: Compute STFT and power spectrogram
    stft = tf.signal.stft(
        waveform,
        frame_length=int(FRAME_SIZE * SAMPLE_RATE),
        frame_step=int(FRAME_STRIDE * SAMPLE_RATE),
        fft_length=NFFT,
        window_fn=tf.signal.hamming_window,
    )
    power_spectrogram = tf.square(tf.abs(stft)) / NFFT

    # * experimenting
    # visualize_spectrogram(power_spectrogram.numpy())
    # *

    # Step 5: Apply Mel filterbanks
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=stft.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=SAMPLE_RATE / 2,
    )
    mel_spectrogram = tf.tensordot(power_spectrogram, mel_filterbank, axes=[-1, 0])
    mel_spectrogram.set_shape(power_spectrogram.shape[:-1] + [NUM_MEL_BINS])

    # Step 6: Convert to log scale
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # * experimenting
    # Add to your pipeline:
    # visualize_log_mel_spectrogram(log_mel_spectrogram.numpy())
    # * experimenting

    # Step 7: Compute MFCCs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :NUM_MFCCS]  # Keep only the first NUM_MFCCS coefficients

    # * experimenting
    # Add to your pipeline:

    # visualize_mfccs(mfccs.numpy())
    # * experimenting

    return mfccs, label


def debug_preprocessing(file_path, label):
    mfccs, label = preprocess_audio(file_path, label)
    print(f"MFCC shape: {mfccs.shape}, Label: {label}")
    return mfccs, label


def prepare_speech_commands_dataset(
    data_dir, batch_size=32, validation_split=0.2, seed=123
):
    """
    Prepare the Speech Commands dataset for training and validation.
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for training and validation.
        validation_split (float): Proportion of the data for validation.
        seed (int): Random seed for shuffling and splitting.
    Returns:
        tuple: (training dataset, validation dataset, list of class names).
    """
    data_dir = Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    file_paths = []
    labels = []

    # Collect all file paths and corresponding labels
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for file_path in class_dir.glob("*.wav"):
            file_paths.append(str(file_path))
            labels.append(label)

    # Convert file paths and labels to TensorFlow constants
    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels)

    # Shuffle and split dataset
    dataset_size = len(file_paths)
    val_size = int(dataset_size * validation_split)

    indices = np.arange(dataset_size)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_file_paths = tf.gather(file_paths, train_indices)
    train_labels = tf.gather(labels, train_indices)

    val_file_paths = tf.gather(file_paths, val_indices)
    val_labels = tf.gather(labels, val_indices)

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_file_paths, val_labels))

    # * trying experimenting here

    # # Add a debugging step to verify statistics
    # train_ds = train_ds.map(
    #     lambda x, y: verify_statistics(x, y), num_parallel_calls=tf.data.AUTOTUNE
    # )

    # train_ds = train_ds.map(lambda x, y: debug_preprocessing(x, y))

    # for mfccs, label in train_ds.take(1):
    #     print(f"MFCC Shape: {mfccs.shape}, Label: {label.numpy()}")
    #     visualize_mfccs(mfccs[0].numpy())

    # * trying experimenting here

    # Apply preprocessing
    train_ds = train_ds.map(
        lambda x, y: preprocess_audio(x, y), num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.map(
        lambda x, y: preprocess_audio(x, y), num_parallel_calls=tf.data.AUTOTUNE
    )

    # Add channel dimension
    train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))

    val_ds = val_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))

    # Apply padded batching
    train_ds = train_ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [None, NUM_MFCCS, 1],
            [],
        ),  # Pad MFCC sequences to [None, NUM_MFCCS] and labels as scalars
        padding_values=(0.0, 0),  # Pad MFCCs with 0.0 and labels with 0
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None, NUM_MFCCS, 1], []),
        padding_values=(0.0, 0),
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# def verify_statistics(file_path, label):
#     mfccs, label = preprocess_audio(file_path, label)
#     print(
#         f"MFCC Stats - Min: {tf.reduce_min(mfccs).numpy()}, Max: {tf.reduce_max(mfccs).numpy()}"
#     )
#     return mfccs, label
