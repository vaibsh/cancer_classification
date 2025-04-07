import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

def fine_tune(model, tokenizer, train_dataset, val_dataset, test_dataset, output_dir,
              batch_size=8, num_epochs=3, weight_decay=0.01, learning_rate=5e-5, use_cuda=False):
    """
    Fine-tune a HuggingFace model with training/validation datasets.

    Args:
        model: HuggingFace model to fine-tune.
        tokenizer: Associated tokenizer.
        train_dataset: Dataset used for training.
        val_dataset: Dataset used for evaluation.
        test_dataset: Dataset used for final evaluation.
        output_dir (str): Directory to save fine-tuned model and logs.
        batch_size (int): Batch size for training and evaluation.
        num_epochs (int): Total number of training epochs.
        weight_decay (float): L2 regularization strength.
        learning_rate (float): Learning rate.
        use_cuda (bool): Whether to use GPU (CUDA) for training.

    Returns:
        model: Fine-tuned HuggingFace model.
        metrics (dict): Evaluation metrics on test set.
    """

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        no_cuda=not use_cuda
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final evaluation on test set
    metrics = trainer.evaluate(test_dataset)
    return model, metrics