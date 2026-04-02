# LLM Prompting & Transformer Fine-Tuning for Text Classification

A comparative NLP study that benchmarks four classification approaches (logistic regression, fine-tuned transformer, zero-shot LLM, and few-shot LLM) on a Tang Dynasty poetry dataset, alongside few-shot LLM poem generation evaluated by a fine-tuned classifier.

## Overview

This project explores the full spectrum of modern NLP classification techniques on a literary dataset of 97 bilingual Tang Dynasty poems. Each poem is labeled as **nature-themed** (landscape, seasons, moon, etc.) or **other**. The project then extends into generative AI, using few-shot prompting to generate synthetic poems and evaluating them with the fine-tuned classifier.

**Key question:** How do prompt engineering, fine-tuning, and classical ML compare on a small, domain-specific text classification task?

---

## Methods

### 1. Baseline — Logistic Regression (Bag-of-Words)
- `CountVectorizer` (max 5,000 features) + `LogisticRegression`
- Stratified 80/20 train/test split
- Evaluated with precision, recall, F1

### 2. Fine-Tuned Transformer
- Model: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- Fine-tuned for binary sequence classification using HuggingFace `Trainer`
- Custom `PoemDataset` PyTorch class wrapping tokenized encodings
- Training args: 3 epochs, batch size 8, lr 5e-5, weight decay 0.01, warmup 25 steps
- Best model checkpoint selected by weighted F1 on eval set
- Saved for reuse in poem generation evaluation

### 3. Zero-Shot LLM Classification
- Model: `gemma3:1b-it-qat` (served locally via Ollama, accessed through OpenAI-compatible API)
- Custom `OllamaChatModel` wrapper class
- **Two prompt variants compared:**
  - *Specific prompt:* Detailed task description, constrained output format ("respond with exactly one lowercase word")
  - *Vague prompt:* Minimal instructions, open-ended output format
- Validity tracking: measures what % of outputs conform to expected format

### 4. Few-Shot LLM Classification
- Custom `FewShotPromptBuilder` class with parameterized prompt and example templates
- Balanced example curation from training set (5 natural + 5 other poems)
- Benchmarked across `num_examples` ∈ {1, 3, 5}
- Results compared in a summary DataFrame

### 5. Few-Shot Poem Generation
- `FewShotPromptBuilder` initialized separately for "NATURAL" and "OTHER" example pools
- Generated 10 poems per class (20 total)
- Style-constrained prompting: classical imagery, no rhyme, 1–3 lines, no modern language
- Generated poems evaluated by the fine-tuned transformer classifier

---

## Key Findings

- **Prompt specificity matters enormously:** The specific zero-shot prompt achieved 100% output validity and substantially higher F1; the vague prompt collapsed into predicting a single label with ~20% valid outputs
- **Fine-tuned transformer outperformed LLM prompting** despite being a far smaller model — demonstrating that fine-tuning on even a small labeled dataset beats zero/few-shot on domain-specific tasks
- **Few-shot improves over zero-shot**, but gains diminish as `num_examples` grows, suggesting the model saturates quickly given limited context length
- **LLMs are not classifiers by default:** without careful output constraints, models default to semantic priors and irregular formatting

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `transformers` (HuggingFace) | Model loading, tokenization, fine-tuning |
| `torch` | Custom dataset, inference |
| `openai` (Python SDK) | Ollama API wrapper |
| `scikit-learn` | Baseline model, metrics |
| `pandas` / `numpy` | Data handling |
| `tqdm` | Inference progress tracking |

---

## Setup & Usage

**Requirements**
```
torch
transformers
openai
scikit-learn
pandas
numpy
tqdm
```

Install:
```bash
pip install torch transformers openai scikit-learn pandas numpy tqdm
```

**Ollama (for LLM classification/generation)**
```bash
# Install Ollama, then pull the model
ollama pull gemma3:1b-it-qat
ollama serve  # start the local server
```

**Run the notebook**

Open ipynb file and run all cells sequentially.

---

## Relevance to ML Engineering

This project directly demonstrates:

- **LLM integration** — wrapping a locally-served LLM behind an OpenAI-compatible API, a common production pattern
- **Prompt engineering** — designing, comparing, and iterating on zero-shot and few-shot prompts with structured output constraints
- **Transformer fine-tuning** — end-to-end supervised fine-tuning with HuggingFace `Trainer`, custom datasets, and checkpoint management
- **LLM evaluation** — validity tracking, format compliance, and comparative F1 benchmarking across methods
- **Data pipeline design** — train/test splitting with stratification, leakage prevention, and few-shot example curation from training data only
- **Generative AI evaluation** — using a classifier to score LLM-generated outputs against held-out ground truth
