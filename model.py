from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

def load_model_and_tokenizer(token, model_name="nlpie/tiny-biobert", device="cpu"):
    """
    Logs into Hugging Face, loads tokenizer and model

    Args:
        token (str): Hugging Face access token.
        model_name (str): Hugging Face model name.
        device (str): 'cpu' or 'cuda'.

    Returns:
        model, tokenizer: Loaded model, tokenizer.
    """
    # Login to Hugging Face
    login(token=token)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    return model, tokenizer
