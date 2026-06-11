# LEMoE Query Distiller

This directory contains the training notebook for the **LEMoE Query Distiller**. This model is built on top of the `microsoft/mdeberta-v3-base` architecture and is trained using **Token Classification** (Binary Named Entity Recognition). 

The goal of this model is to act as a precision filter: it evaluates every single word in a user's prompt and classifies it as either `KEEP` (1) or `DISCARD` (0). This effectively strips away noise, stop words, and conversational filler, leaving only the core search intent.

## Expected Data Format

The notebook expects the training dataset to be in **JSONL** (JSON Lines) format. Because this is a token classification task, the data must be pre-tokenized into lists of words, mapped exactly to a list of numerical tags.

- `tokens`: A list of strings, where each string is a word in the sentence.
- `ner_tags`: A list of integers (0 or 1) corresponding to each word. `0` means DISCARD, `1` means KEEP.

### Example `dataset.jsonl`
```json
{"tokens": ["quiero", "ver", "el", "recibo", "de", "la", "luz"], "ner_tags": [0, 0, 0, 1, 0, 0, 1]}
{"tokens": ["búscame", "las", "auditorías", "del", "sistema"], "ner_tags": [0, 0, 1, 0, 1]}
```
*(The notebook automatically generates a sample `dataset.jsonl` file so you can run the code immediately without needing your own data).*

## Pipeline Overview

The notebook is divided into the following key sections:

1. **Dependency Installation**: Installs all required libraries (`transformers`, `datasets`, `seqeval`, `optimum`, etc.).
2. **Sample Data Creation**: Automatically generates a tiny `.jsonl` dataset to make the notebook plug-and-play.
3. **Dataset Loading**: Uses the Hugging Face `datasets` library to load the JSONL file and map the binary labels.
4. **Tokenization and Label Alignment**: The most critical step. Since DeBERTa uses a tokenizer that can split single words into multiple sub-tokens (e.g., `auditorías` -> `audito`, `##rías`), the notebook aligns the original `ner_tags` to the new sub-tokens. Secondary sub-tokens are assigned a label of `-100` so the loss function ignores them.
5. **Model Configuration and Training**: Initializes the DeBERTa token classification architecture and performs the fine-tuning process optimized for small to medium datasets.
6. **Export to ONNX Format**: Converts the fine-tuned model to an ONNX graph to drastically improve inference times in production.
7. **Production Inference Script**: Simulates a production environment. It provides a highly optimized Python script using pure `torch.no_grad()` and dictionary mapping to reconstruct the distilled query in milliseconds.

## How to Use

> **Hardware Note:** The original fine-tuning for this model was executed on a laptop with an NVIDIA RTX 5070. The notebook is fully compatible with cloud platforms like **Google Colab**.

1. Run the first few cells to install dependencies and generate the dummy dataset.
2. If you have your own data, replace the `dataset.jsonl` file with your custom tokenized dataset.
3. Run the rest of the notebook.
4. Your production-ready model will be saved in the `./deberta_finetuned` and `./deberta_onnx` folders, ready to be integrated into your semantic search or RAG pipeline.
