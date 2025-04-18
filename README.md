# Cancer Classifier
Research papers from PubMed are provided with PubMedId, Tile and Abstract for each paper. The task is to 
classify the papers into Cancer or Non-Cancer categories

## Overview

- Parses PubMed abstracts and title from the dataset. Pre-processes the parsed data
- Uses pre-trained tiny-biobert model (via Hugging Face Transformers) for classification.
- Data is split into train, validation and test datasets in the ratio of 70% train : 10% validation : 20% test
- The model is fine-tuned on train dataset. Grid search is used to find optimal hyper-params viz. batch_size, num_epochs, weight_decay and 
  learning_rate. Information on these hyper-params can be found here: https://huggingface.co/docs/transformers/en/main_classes/trainer
- Performance of the model is computed before and after fine-tuning. Predictions are made on the fine-tuned model
- Note: For model fine-tuning and testing, text data was used as Abstract only. Experiments were conducted with Abstract only, Title only,
  Title + ' .' + Abstract . Out of these three possibilities, Abstracts only gave the highest performance on the data provided

## Choice of Model
Model chosen is nlpie/tiny-biobert https://huggingface.co/nlpie/tiny-biobert
Reasons for choosing this model
- tiny-biobert is "tiny" as it is distilled version of BioBERT. It is less compute and memory intensive and hence can be run on local CPUs
- Like BioBERT, tiny-biobert is pre-trained on PubMed abstracts and hence it understands domain medical jargon like "carcinoma", "oncology", "benign" etc 
- Other models were considered like PubMedBERT, but I found distilled version of BioBERT i.e. tiny-biobert readily available on HuggingFace

## Project Structure
<pre>
cancer_classification/
├── main.py                         # Main: Specify hugging face token, input and output paths
├── parser.py                       # Loads and parses input data
├── model.py                        # Loads tokenizer and pre-trained model
├── dataset_preparation.py          # Prepares dataset for training
├── train.py                        # Fine-tunes the model
├── grid_search.py                  # Runs grid search for best hyperparams
├── eval.py                         # Evaluates model performance
├── predict.py                      # Adds predictions to the dataset
├── requirements.txt                # Dependencies
│
├── notebooks/
│   └── model.ipynb                 # Jupyter notebook for experimentation
│
├── results/
│   ├── base_model_performance.txt         # Results from base model
│   ├── fine_tuned_model_performance.txt   # Results from fine-tuned model
│   └── classified_data.csv                # Final predictions with probabilities
│
├── Dataset/
│   ├── Cancer/                     # Cancer-related PubMed abstracts (text files)
│   └── Non-Cancer/                 # Non-Cancer-related PubMed abstracts
│
└── README.md                       # Project overview and usage
</pre>

## Usage
<pre>
pip install -r requirements.txt
python main.py              # Execute the code and save the results at the specified path
</pre>
  
## Future Work
- Experiment with other models and with greater hardware resources like GPU to find out the best model for this problem statement
- Text data provided in the form of Abstracts, Title must be experimented with further in terms of pre-processing. For instance, does
  removing tags like "OBJECTIVE:", "BACKGROUND:", "IMPORTANCE:" have any impact on performance. Does combining title and abstract 
  information have any impact on performance etc 
- There are 1000 data points in the input dataset, 70% of which are used for fine-tuning. This works well since the model is few-shot 
  learner. But, performance gains, if any must be experimented with more data points
- Misclassified PubMed articles must be manually inspected for any words/terms to find out why the articles are getting misclassified
