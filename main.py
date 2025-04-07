import torch
from parser import parse_text_files
from model import load_model_and_tokenizer
from dataset_preparation import prepare_datasets
from eval import evaluate_model
from grid_search import run_grid_search
from predict import add_predictions


def main():
    # Parse text files
    df = parse_text_files(base_path="Dataset")  # Insert input dataset here

    # Load model and tokenizer
    token = "<token>"           # Insert your HuggingFace token here
    model_name = "nlpie/tiny-biobert"
    device = torch.device("cpu")
    model, tokenizer = load_model_and_tokenizer(token, model_name, device)

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(df, tokenizer)

    # Evaluate base model
    evaluate_model(model, tokenizer, df, device, output_path='results/base_model_performance.txt')

    # Grid search to find best fine-tuned model
    best_params, best_model = run_grid_search(
            model, tokenizer, train_dataset, val_dataset, test_dataset
    )

    # Evaluate fine-tuned model
    evaluate_model(best_model, tokenizer, df, device, output_path='results/fine_tuned_model_performance.txt')

    # Run final predictions
    predictions_df = add_predictions(df, best_model, tokenizer, device, output_path='results/classified_data.csv')
    print(predictions_df.head())


if __name__ == "__main__":
    main()