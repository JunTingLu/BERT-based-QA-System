# 🧠 Fine-Tuning LLM with BERT

This project demonstrates how to fine-tune **BERT** on a custom **Question & Answer (QA)** dataset using **Google Colab with a T4 GPU**. It’s designed for engineers and researchers who want to build QA models quickly and efficiently without local GPU setup.

This project frames the QA task as a Masked Language Modeling (MLM) problem, where the model learns to predict the answer by filling a `[MASK]` token.

## 💻 Environment
| Platform          | GPU Type | Runtime     | Notes                                    |
|--------------------|----------|-------------|-------------------------------------------|
| Google Colab       | T4       | Python 3.10+ | Recommended for quick training sessions |

## 📌 Features
- ✅ Fine-tune BERT using Hugging Face Transformers
- 🚀 Optimized for Google Colab T4 GPU
- 📝 Supports QA-formatted datasets
- 🧪 Automated data preprocessing and training flow
- 📈 Loss tracking and performance monitoring
- 🧠 Easily extendable to multiple-choice and multilingual tasks

---

## 🧠 Model Details
- Base model: bert-base-chinese
- Tokenizer: AutoTokenizer
- Training objective: Masked Language Modeling (QA Fine-tuning)
- Output: A fine-tuned BERT model ready for inference

---
## 📂 Project Structure
```text
├── fine_tuning_llm_round1.ipynb # Main Jupyter notebook (data processing, training & evaluation)
├── data/ # QA datasets (CSV / JSON)
│   └── train.csv
├── models/ # Saved fine-tuned model
└── README.md # Project documentation
```
---
## DeepSpeed
Fine-tune Bert-base-Chinese models efficiently using DeepSpeed, with support for ZeRO optimization and mixed precision.

### ⚡Setup
1. **Create a Conda environment**:
```bash
conda create -n deepspeed_env python=3.12 -y
conda activate deepspeed_env
```
2. **Install dependencies**
  - Install PyTorch (choose the correct CUDA version from PyTorch website)
  - Install DeepSpeed and Transformers:
    ```bash
    pip install deepspeed transformers datasets
    ```
3. **Dataset** 
Make sure your dataset follows the format below (e.g. `data/train.csv`):
```csv
question,answer
What is the capital of France?,Paris
Who founded Microsoft?,Bill Gates
```
---
4. **Fine-tuning**:
Execute the `run_qa.py` script with DeepSpeed to start fine-tuning. The following command fine-tunes `bert-base-chinese` for 3 epochs.

```bash
deepspeed --num_gpus=1 run_qa.py \
    --model_name_or_path bert-base-chinese \
    --train_file data/train.csv \
    --output_dir models/bert-qa-chinese \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --fp16
``` 
5. **Interact with the model**:
After training, you can interact with the fine-tuned model using the `fill-mask` pipeline from Hugging Face Transformers. The model will predict the answer to your question by filling in the `[MASK]` token.

```python
from transformers import pipeline

# Load the fine-tuned model from the output directory
qa_pipeline = pipeline(
    "fill-mask",
    model="models/bert-qa-chinese",
    tokenizer="models/bert-qa-chinese"
)

# Format the question for the fill-mask pipeline
question = "What is the capital of France?"
result = qa_pipeline(f"{question} [MASK]")

# The top prediction is the answer
print(f"Question: {question}")
print(f"Answer: {result[0]['token_str']}")
```

## 📊 Training Tips
If the loss does not decrease:
- Check if labels are aligned with the answers
- Disable fp16 if you encounter unstable training
- Ensure there are no missing or malformed rows in the dataset
---
## References
- [DeepSpeed](https://www.deepspeed.ai)
- [DeepSpeed introuduction](https://zhuanlan.zhihu.com/p/690690979)



