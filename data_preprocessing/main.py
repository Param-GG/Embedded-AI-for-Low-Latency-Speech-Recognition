import sys
from dataset import prepare_speech_commands_dataset

import os
import logging
import tensorflow as tf

# Suppress TensorFlow logs below the error level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').addHandler(logging.NullHandler())

# Optional: Ensure CUDA-related logs are also suppressed
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Make sure only your target GPU is used
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Helps in blocking and minimizing verbosity


def main():
    
    data_dir = 'datasets/speech_commands_v0_extracted'
    train_ds, val_ds, test_ds, class_names = prepare_speech_commands_dataset(data_dir)
    
    # Display some information about the dataset
    print("Class Names:", class_names)
    print("Training dataset sample:")
    print("Detected Class Names:", class_names)

    for batch in train_ds.take(1):
        mfccs, labels = batch
        print("MFCC shape:", mfccs.shape)
        print("Labels:", labels)

if __name__ == "__main__":
    main()

