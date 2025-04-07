import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(model, tokenizer, df, device, output_path):
    """
    Evaluates the model on a dataset and saves accuracy, F1 score, and confusion matrix to file.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        df (pd.DataFrame): Input DataFrame with 'Abstract' and 'Category' columns.
        device: Torch device (e.g., 'cuda' or 'cpu').
        output_path (str): File path to save evaluation results.
    """

    def predict(text):
        inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_label = "Cancer" if torch.argmax(logits, dim=1).item() == 1 else "Non-Cancer"
        return pred_label

    # Apply predictions and convert to NumPy arrays
    predicted_categories = df["Abstract"].apply(predict).to_numpy()
    true_categories = df["Category"].to_numpy()

    # Compute metrics
    accuracy = accuracy_score(true_categories, predicted_categories)
    f1 = f1_score(true_categories, predicted_categories, pos_label="Cancer")
    conf_matrix = confusion_matrix(true_categories, predicted_categories)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write results to file
    with open(output_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))

    print(f"Evaluation results saved to {output_path}")