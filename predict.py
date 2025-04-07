import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def add_predictions(df, model, tokenizer, device='cpu'):
    """
    Adds prediction results to a DataFrame with abstracts.

    Args:
        df (pd.DataFrame): DataFrame containing an 'Abstract' column.
        model: HuggingFace model for classification.
        tokenizer: Corresponding tokenizer.
        device (str): 'cpu' or 'cuda'.

    Returns:
        pd.DataFrame: Updated DataFrame with prediction columns.
    """
    model.to(device)
    model.eval()

    def predict(text):
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze()
            pred_label = torch.argmax(probs).item()

        return pd.Series([
            "Cancer" if pred_label == 1 else "Non-Cancer",
            float(probs[0].item()),  # Non-Cancer Score
            float(probs[1].item())   # Cancer Score
        ])

    df[["Predicted_Category", "Non-Cancer Score", "Cancer Score"]] = df["Abstract"].progress_apply(predict)
    return df