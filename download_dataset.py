import tensorflow as tf
from tensorflow.keras.utils import get_file


def download_dataset():
    """Downloads and extracts the TensorFlow Speech Commands dataset."""
    dataset_url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    dataset_path = get_file(
        fname="speech_commands_v0.02",  # Name for the cached file
        origin=dataset_url,
        extract=True,
        cache_dir="./",
        cache_subdir="datasets",
    )

    # The extracted files will be in the "datasets/speech_commands_v0.02" folder
    extracted_path = f"./datasets/speech_commands_v0.02"

    return extracted_path


# Example usage:
dataset_path = download_dataset()
print(f"Dataset downloaded and extracted to: {dataset_path}")
