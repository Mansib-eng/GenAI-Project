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
│   ├── GenAi.md
│   ├── GenAi.pdf
│   └── notebooks/
│       └── 01_environment_test.ipynb
│
├── project_01_classical_baseline/
│   ├── data/
│   │   ├── sample_raw_texts.txt
│   │   └── sample_cleaned_texts.txt
│   ├── notebooks/
│   │   └── 01_text_cleaning_test.ipynb
│   └── src/
│       └── text_cleaning.py
│
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


## 10. Make your first Git commit

Check the repository:

```powershell
git status
```

