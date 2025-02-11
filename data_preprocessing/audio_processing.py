import random
import tensorflow as tf
from pathlib import Path
from utils import load_config

# Load configuration parameters
config = load_config()
SAMPLE_RATE = config["sample_rate"]
FRAME_SIZE = config["frame_size"]
FRAME_STRIDE = config["frame_stride"]
NFFT = config["nfft"]
NUM_MEL_BINS = config["num_mel_bins"]
NUM_MFCCS = config["num_mfccs"]
BACKGROUND_NOISE_DIR = config["background_noise_dir"]

def add_background_noise(audio):
    """
    Adds background noise to the audio signal.
    
    Args:
        audio (tf.Tensor): Input audio signal (1D tensor).
        
    Returns:
        tf.Tensor: Augmented audio with background noise.
    """
    # Get all background noise files
    noise_files = list(Path(BACKGROUND_NOISE_DIR).glob("*.wav"))
    if not noise_files:
        return audio  # If no noise files found, return original audio

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
    noise_waveform = noise_waveform[: tf.shape(audio)[0]]  # Trim to match target audio length

    # Scale background noise to 10% of the target audio's RMS volume
    audio_rms = tf.math.sqrt(tf.reduce_mean(tf.square(audio)))
    noise_rms = tf.math.sqrt(tf.reduce_mean(tf.square(noise_waveform)))
    noise_waveform = noise_waveform * (0.1 * audio_rms / noise_rms)

    # Add the scaled background noise to the target audio
    augmented_audio = audio + noise_waveform

    return augmented_audio

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

    # Step 4: Compute STFT and power spectrogram
    stft = tf.signal.stft(
        waveform,
        frame_length=int(FRAME_SIZE * SAMPLE_RATE),
        frame_step=int(FRAME_STRIDE * SAMPLE_RATE),
        fft_length=NFFT,
        window_fn=tf.signal.hamming_window,
    )
    power_spectrogram = tf.square(tf.abs(stft)) / NFFT

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

    # Step 7: Compute MFCCs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :NUM_MFCCS]  # Keep only the first NUM_MFCCS coefficients

    return mfccs, label

def debug_preprocessing(file_path, label):
    """
    A helper function to print the shape of the MFCCs for debugging.
    """
    mfccs, label = preprocess_audio(file_path, label)
    print(f"MFCC shape: {mfccs.shape}, Label: {label}")
    return mfccs, label
