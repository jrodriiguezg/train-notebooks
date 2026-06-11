# LEMoE PPC (Pipeline for Classification)

This directory contains the training notebook for the **LEMoE PPC** model. This is a sequence classification model built on top of the `dccuchile/bert-base-spanish-wwm-cased` architecture (or similar BERT variants), designed to categorize, filter, or route incoming text queries.

# Expected Data Format

The notebook expects the training dataset to be in **JSONL** (JSON Lines) format. Each line in the file must be a valid JSON object containing at least two fields:
- `text`: The input string / user query.
- `label`: An integer representing the target class (e.g., `0` for General/Informative, `1` for Filter/Documentation).

### Example `papper_data.jsonl`
```json
{"text": "¿Cómo instalo el servidor?", "label": 1}
{"text": "Hola, buenos días", "label": 0}
{"text": "Necesito la auditoría del mes pasado", "label": 1}
{"text": "¿Qué hora es?", "label": 0}
```

##  Pipeline Overview

The notebook is divided into the following key sections:

1. **Dependency Installation**: Automatically installs `torch`, `transformers`, `datasets`, `optimum`, and `onnxruntime`.
2. **Dataset Preparation**: Loads the custom `.jsonl` file efficiently using Hugging Face's `datasets` library.
3. **Tokenization and Model Configuration**: Initializes the BERT tokenizer and configures the sequence classification model with the appropriate number of labels.
4. **Fine-Tuning Process**: Configures training parameters (learning rate, batch size) and leverages **fp16 mixed precision** to train the model quickly.
5. **Live Inference Interface**: Provides an interactive chat loop right in the notebook so you can test the model's predictions in real-time.
6. **Export to ONNX Format**: Converts the trained PyTorch model into an ONNX static graph. This is critical for deploying the model in production environments without the heavy PyTorch overhead.
7. **Dynamic INT8 Quantization**: Applies CPU-optimized quantization to the ONNX model, significantly reducing the file size and accelerating inference speed with minimal accuracy loss.
8. **Verification**: A final sanity check to ensure the quantized ONNX model produces the correct logits.

##  How to Use

> **Hardware Note:** The original fine-tuning for this model was executed on a laptop with an NVIDIA RTX 5070. The code is highly optimized (using fp16 mixed precision) and can be easily run on **Google Colab** without any issues.

1. Place your training data in the same directory and name it `papper_data.jsonl` (or update the filename in the notebook).
2. Run the notebook from top to bottom.
3. Once finished, you will find your deployment-ready models inside the generated `./output`, `./modelo_onnx`, and `./modelo_onnx_cuantizado` folders.
