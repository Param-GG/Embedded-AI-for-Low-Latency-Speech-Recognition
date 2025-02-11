import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing.dataset_handling import prepare_speech_commands_dataset

def evaluate_model(model, test_ds, class_names):
    """
    Evaluate model performance on test dataset.
    """
    # Get predictions
    y_pred = []
    y_true = []
    
    for features, labels in tqdm(test_ds, desc="Evaluating"):
        predictions = model.predict(features, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        y_pred.extend(pred_labels)
        y_true.extend(labels.numpy())

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load the saved model
    model = tf.keras.models.load_model("ds_cnn_model.h5")
    
    # Load test dataset
    data_dir = 'datasets/speech_commands_v0_extracted'
    batch_size = 32
    _, _, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)
    
    # Evaluate
    evaluate_model(model, test_ds, class_names)

if __name__ == "__main__":
    main()