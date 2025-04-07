import os
from sklearn.model_selection import ParameterGrid
from train import fine_tune

def run_grid_search(model, tokenizer, train_dataset, val_dataset, test_dataset, base_output_dir="results/grid_search", use_cuda=False):
    """
    Run grid search for hyperparameter tuning.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        test_dataset: Dataset for evaluation
        base_output_dir: Directory to save models and logs
        use_cuda: Whether to use GPU

    Returns:
        best_params: Dict of best hyperparameters
        best_model: Best model after grid search

    """

    param_grid = {
        "batch_size": [8, 16],
        "num_epochs": [2, 3, 5],
        "weight_decay": [0.0, 0.01, 0.1],
        "learning_rate": [5e-5, 3e-5, 2e-5]
    }

    """
    param_grid = {
        "batch_size": [8],
        "num_epochs": [1],
        "weight_decay": [0.01],
        "learning_rate": [5e-5]
    }
    """

    best_f1 = 0
    best_params = None
    best_model = None

    os.makedirs(base_output_dir, exist_ok=True)

    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")

        output_dir = os.path.join(base_output_dir, f"bs{params['batch_size']}_ep{params['num_epochs']}_wd{params['weight_decay']}_lr{params['learning_rate']}")

        model, metrics = fine_tune(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            output_dir=output_dir,
            batch_size=params["batch_size"],
            num_epochs=params["num_epochs"],
            weight_decay=params["weight_decay"],
            learning_rate=params["learning_rate"],
            use_cuda=use_cuda
        )

        f1 = metrics.get("eval_f1", 0)
        print(f"F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model

    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Parameters: {best_params}")

    return best_params, best_model