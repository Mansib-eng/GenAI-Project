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

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```


-----

## Project 1: Reproducible Classical Baseline

### Task 1: Reusable text-cleaning functions

The text-cleaning module performs:

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
GenAI-Project/
└── project_01_classical_baseline/
    ├── data/
    │   └── sample_raw_texts.txt
    ├── notebooks/
    │   └── 01_text_cleaning_test.ipynb
    └── src/
        └── text_cleaning.py

```

-----


## 10. Make your first Git commit

Check the repository:

```powershell
git status
```

