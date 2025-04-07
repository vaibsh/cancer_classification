from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset

def prepare_datasets(df, tokenizer):
    """
    Maps labels, splits the data, tokenizes texts, and returns HuggingFace datasets.

    Args:
        df (pd.DataFrame): DataFrame with 'Abstract' and 'Category' columns.
        tokenizer: HuggingFace tokenizer.

    Returns:
        train_dataset, val_dataset, test_dataset (HuggingFace Datasets)
    """
    # Map categories to labels
    label_mapping = {'Non-Cancer': 0, 'Cancer': 1}
    df['Label'] = df['Category'].map(label_mapping)

    # Split data
    train_texts, tmp_texts, train_labels, tmp_labels = train_test_split(
        df['Abstract'], df['Label'], test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        tmp_texts, tmp_labels, test_size=2/3, random_state=42
    )

    # Tokenization
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

    # Convert to HuggingFace datasets
    train_dataset = HFDataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": list(train_labels)
    })
    val_dataset = HFDataset.from_dict({
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": list(val_labels)
    })
    test_dataset = HFDataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": list(test_labels)
    })

    return train_dataset, val_dataset, test_dataset