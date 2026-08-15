# GenAI Learning

This repository documents my structured journey in Generative AI.

## Learning objectives

- Understand Generative AI and large language models
- Learn PyTorch and the Hugging Face ecosystem
- Work with LLM APIs and local models
- Build retrieval-augmented generation systems
- Explore parameter-efficient fine-tuning
- Evaluate and deploy GenAI applications

## Project structure

- `notebooks/` - Learning exercises and experiments
- `src/` - Reusable Python code
- `data/` - Local datasets not tracked by Git
- `requirements.txt` - Python dependencies

```
GenAI-Project/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── week_01_foundations/
│   ├── GenAi.pdf
│   └── notebooks/
│       └── 01_environment_test.ipynb
│
├── project_01_classical_baseline/
│   ├── data/
│   │   ├── sample_raw_texts.txt
│   │   ├── sample_cleaned_texts.txt
|   |   └── practice_clickbait.csv
|   ├── models/
│   │   └── project_01_classical_baseline.joblib
│   ├── notebooks/
│   |   ├── 01_text_cleaning_test.ipynb
│   |   ├── 02_numpy_vectors.ipynb
│   |   ├── 03_pandas_data_quality.ipynb
│   |   ├── 04_ml_workflow.ipynb
│   |   ├── 05_classical_baseline.ipynb
│   |   ├── 06_evaluation_error_analysis.ipynb
|        └── 07_project_01_checkpoint.ipynb
|   ├── results/
│   |   ├── day_05_baseline_metrics.json
|   |   ├── day_06_error_analysis.md
|   |   ├── project_01_checkpoint_metrics.json
|   |   ├── project_01_test_predictions.csv
|   |   └── project_01_confusion_matrix.png
│   ├── src/
│       └── text_cleaning.py
│   ├── README.md
|   ├── requirements.txt
|   └── week_02_python_data_ml_refresher.pdf

├── project_02_transformer_lab/
│   └── .gitkeep
│
├── project_03_clickbait_analyst/
│   └── .gitkeep
│
├── project_04_rag_assistant/
│   └── .gitkeep
│
├── project_05_lora_study/
│   └── .gitkeep
│
├── project_06_agent_workflow/
│   └── .gitkeep
│
└── final_capstone/
    └── .gitkeep
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```


-----

## Project 1: Reproducible Classical Baseline

### Task 1: Reusable text-cleaning functions

The **text-cleaning module** performs:

- Unicode NFC normalization
- Missing-value handling
- Whitespace normalization
- URL replacement using `<URL>`
- Hashtag preservation using `<HASHTAG> hashtag_word`
- Repeated-punctuation limiting
- Exact duplicate removal
- UTF-8 file reading and writing

Bengali text, English text, code-mixed content, and emojis are preserved.

```
├── project_01_classical_baseline/
│   ├── data/
│   │   ├── sample_raw_texts.txt
│   │   └── sample_cleaned_texts.txt
│   ├── notebooks/
│   │   └── 01_text_cleaning_test.ipynb
│   └── src/
│       └── text_cleaning.py

```

-----

### Task 2: Calculate similarities manually and with NumPy

```
GenAI-Project/
└── project_01_classical_baseline/
    └── notebooks/
        ├── 01_text_cleaning_test.ipynb
        └── 02_numpy_vectors.ipynb
```

**02_numpy_vectors.ipynb** covers:

- NumPy arrays, dimensions, and shapes
- Manual dot-product calculation
- NumPy dot product
- Manual vector magnitude
- Manual cosine similarity
- NumPy cosine similarity
- Reusable similarity functions
- Text represented as bag-of-words vectors
- Pairwise similarity matrix
- Zero-vector error handling
- Practice exercise and automatic assertions

-----

### Task 3: Produce a compact data-quality report

```
GenAI-Project/
└── project_01_classical_baseline/
    └── notebooks/
        ├── 01_text_cleaning_test.ipynb
        ├── 02_numpy_vectors.ipynb
        └── 03_pandas_data_quality.ipynb
```

**03_pandas_data_quality.ipynb** covers:

- Loading CSV data using pandas
- Inspecting shape, columns, data types, and missing values
- Filtering valid records
- Grouping by label and platform
- Detecting missing or empty text
- Detecting duplicate text
- Finding invalid labels and unexpected platforms
- Producing a compact data-quality report
- Automated verification with assertions
- Independent practice exercises

-----

### Task 4: Build a clean train/validation/test pipeline

```
project_01_classical_baseline/
└── notebooks/
    ├── 01_text_cleaning_test.ipynb
    ├── 02_numpy_vectors.ipynb
    ├── 03_pandas_data_quality.ipynb
    └── 04_ml_workflow.ipynb

```

**04_ml_workflow.ipynb** teaches:

- Train / validation / test split
- Why we need three datasets
- Stratified splitting
- What data leakage is
- How TF-IDF can cause leakage
- `DummyClassifier` baseline
- TF-IDF + Logistic Regression
- `Pipeline`
- Training vs validation performance
- Detecting possible overfitting
- Final test evaluation
- Why the test set should remain untouched
- Automatic assertions
- Independent exercises

The notebook produces the required evidence: a clean, leakage-safe train/validation/test ML pipeline.

------


### Task 5: Train a clickbait baseline and save metrics

```
project_01_classical_baseline/
├── notebooks/
│   ├── 01_text_cleaning_test.ipynb
│   ├── 02_numpy_vectors.ipynb
│   ├── 03_pandas_data_quality.ipynb
│   ├── 04_ml_workflow.ipynb
│   └── 05_classical_baseline.ipynb
├── results/
│   └── day_05_baseline_metrics.json
└── src/
    └── text_cleaning.py

```

**05_classical_baseline.ipynb** is essentially the first complete classical ML text-classification experiment.

The final baseline achieved:

```
Accuracy:   0.833
Precision:  0.750
Recall:     1.000
F1:         0.857
Macro-F1:   0.829

```



| Part | What learn |
|---|---|
| Dataset | Creates factual (`0`) and clickbait (`1`) text examples |
| Stratified split | Creates 70% train, 15% validation, 15% test while maintaining class balance |
| TF-IDF | Converts text into numerical features |
| Logistic Regression | Learns to classify factual vs clickbait |
| `(1,1)` | Tests unigrams |
| `(1,2)` | Tests unigrams + bigrams |
| `(1,3)` | Tests unigrams + bigrams + trigrams |
| Validation | Compares the three TF-IDF configurations |
| Model selection | Selects the best configuration using validation Macro-F1 |
| Test evaluation | Tests the selected model on unseen data |
| Accuracy | Overall percentage of correct predictions |
| Precision | When model says "clickbait", how often it is correct |
| Recall | How many actual clickbait examples it finds |
| F1 | Balance between precision and recall |
| Macro-F1 | Gives equal importance to each class |
| JSON saving | Saves final experiment results to `day_05_baseline_metrics.json` |


-----


### Task 6: Evaluation error-analysis 

```
project_01_classical_baseline/
├── notebooks/
│   ├── 05_classical_baseline.ipynb
│   └── 06_evaluation_error_analysis.ipynb
└── results/
    ├── day_05_baseline_metrics.json
    └── day_06_error_analysis.md

```

Key Findings:

```
Accuracy:       0.833
Macro-F1:       0.829

True negatives:  2
False positives: 1
False negatives: 0
True positives:  3

```

The only error was:

```
Researchers publish results of health study
```

**06_evaluation_error_analysis.ipynb** covers:

- Accuracy and Macro-F1
- Confusion matrix
- True positives, true negatives, false positives, and false negatives
- Precision, recall, F1, and support for each class
- Test-prediction inspection
- Misclassified-example analysis
- TF-IDF vocabulary limitations
- Prediction probability interpretation
- One-page error-analysis generation

-----

### Day 7: Project 1 Checkpoint

**Notebook:** `07_project_01_checkpoint.ipynb`

The final checkpoint combines the complete workflow:

- Loading and checking the dataset
- Creating stratified data splits
- Comparing TF-IDF configurations
- Selecting the best model
- Evaluating the model on unseen test data
- Saving the trained model
- Exporting predictions, metrics, and the confusion matrix

#### Final Checkpoint Results

```text
Selected model:          Word TF-IDF (1,1) + Logistic Regression
Validation Macro-F1:     0.829
Test accuracy:           0.833
Test Macro-F1:           0.829
```

#### Final Project Structure

```text
project_01_classical_baseline/
├── data/
│   └── practice_clickbait.csv
├── notebooks/
│   ├── 05_classical_baseline.ipynb
│   ├── 06_evaluation_error_analysis.ipynb
│   └── 07_project_01_checkpoint.ipynb
├── models/
│   └── project_01_classical_baseline.joblib
└── results/
    ├── project_01_checkpoint_metrics.json
    ├── project_01_test_predictions.csv
    └── project_01_confusion_matrix.png
```

---


## 10. Make your first Git commit

Check the repository:

```powershell
git status
```

