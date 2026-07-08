# XplorIT — Well-Being Video Classification Pipeline

An applied ML research project classifying self-reflection videos for well-being studies, built during a research partnership (UvA / OceansX). The pipeline turns raw YouTube content into structured training data and benchmarks three classifier families on it.

## What it does

1. **Data collection** — scrapes candidate self-reflection videos from YouTube (BeautifulSoup, Selenium) and transcribes them with OpenAI Whisper.
2. **Data handling** — cleans, filters, and structures transcripts into a labeled training corpus, with iteratively augmented keyword filters.
3. **Classification** — trains and compares three model families on the corpus:
   - BERT (transformer fine-tuning)
   - SVM
   - Random Forest
4. **Interpretation** — applies LIME and SHAP to inspect model reasoning and identify which expressions the models associate with well-being.

## Repo structure

| Folder | Contents |
|---|---|
| `youtube/` | Scraping + transcription pipeline |
| `Data Handling/` | Corpus cleaning and structuring |
| `BERT/` | Transformer fine-tuning notebooks |
| `SVM model/` | SVM training + evaluation |
| `Random forests/` | Random Forest training + evaluation |

## Stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · BeautifulSoup · Selenium · OpenAI Whisper · LIME · SHAP

## Outcome

The final models were hosted on Hugging Face and presented via an interactive demo, which took **1st place** at the project competition.
