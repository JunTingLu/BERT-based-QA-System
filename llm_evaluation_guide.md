# Guide to Evaluating Answer Quality Using the Qwen Model

This project uses the **Qwen-3B model** as an evaluator (Judge) to compare the answer quality of a **fine-tuned BERT model** against the **original baseline model**.  
This evaluation follows the **LLM-as-a-Judge** paradigm.

---

## 🎯 Core Concept

The core idea of **LLM-as-a-Judge** is to leverage a powerful and neutral third-party large language model (in this project, **Qwen**) to automatically evaluate the output quality of other models (BERT before and after fine-tuning).

Traditionally, such evaluations rely on extensive human annotation and subjective scoring.  
LLM-as-a-Judge provides a **faster, more scalable, and cost-efficient alternative**.

The evaluator (Judge) model receives a prompt containing:

1. The original question  
2. A reference **gold answer**  
3. The response from **Model A** (baseline BERT)  
4. The response from **Model B** (fine-tuned BERT)

The Judge then scores both responses based on predefined criteria (correctness, relevance, and completeness) and determines which model performs better.

---

## 🔍 Evaluation Workflow

The complete evaluation pipeline is as follows:

1. **Benchmark Dataset Generation**  
   Randomly sample questions from the `ikala/tmmluplus` dataset to construct a standardized QA benchmark set named `qa_benchmark`.

2. **Model Inference**  
   - **Model A (Baseline)**: Use the original `bert-base-chinese` model to generate answers.  
   - **Model B (Fine-tuned)**: Use the BERT model fine-tuned on `ikala/tmmluplus` to generate answers for the same questions.

3. **Evaluation Prompt Construction**  
   For each question, dynamically build a structured prompt that includes the question, reference answer, and outputs from both models.

4. **Evaluation Execution**  
   Send the constructed prompt to the Qwen model acting as the Judge.

5. **Result Collection**  
   Qwen returns numerical scores, qualitative feedback, and a final verdict indicating the superior model.

---

## 🧩 Code Structure Overview

The project is primarily implemented in `fine_tuning_llm_round1.ipynb`, with the following key components:

- **`generate_qa_benchmark()`**  
  Loads datasets from the Hugging Face Hub and randomly samples QA pairs for evaluation.

- **`get_dataset()`**  
  Loads and preprocesses training data for BERT fine-tuning.

- **Training Loop**  
  Fine-tunes `bert-base-chinese` efficiently using the **DeepSpeed** framework.

- **`chat_with_tuning_llm()`**  
  Interacts with the **fine-tuned** BERT model to generate responses.

- **`general_chat()`**  
  Interacts with the **original** `bert-base-chinese` model to generate baseline responses.

- **`EvalLLm` Class**  
  - `__init__`: Initializes the Judge model (Qwen) and loads gold answers and model outputs.  
  - `evaluation()`: Constructs evaluation prompts and retrieves results from the Judge model.

- **Main Evaluation Loop**  
  Iterates over all questions in `qa_benchmark` and outputs evaluation reports.

---

## 🧠 Prompt Design

To guide the evaluator toward fair and structured judgments, the following prompt template is used:

```python
EVAL_PROMPT = """
[Question]: {question}
[Reference Answer]: {gold_answer}
[Model A Answer]: {baseline_output}
[Model B Answer]: {finetuned_output}

Act as an impartial judge.
Evaluate both Model A and Model B’s answers with respect to correctness, relevance, and completeness.
Must give a score from 1-10 for each, and declare which is better:
<your evaluation here>
"""
```

### Prompt Components

- `{question}`: Original question  
- `{gold_answer}`: Reference (gold) answer  
- `{baseline_output}`: Output from the baseline model  
- `{finetuned_output}`: Output from the fine-tuned model  

---

## ⚙️ Configuration Options

- **`model_name` (in `EvalLLm`)**  
  Default: `Qwen/Qwen2.5-3B-Instruct`  
  Can be replaced with other powerful models such as `gpt-4` or `claude-3-opus`.

- **`config_params` (notebook)**  
  DeepSpeed training parameters including `train_batch_size`, `lr`, etc.

- **`random_seed` (in `generate_qa_benchmark`)**  
  Ensures reproducibility of the benchmark dataset.

---

## 🔬 Technical Details

- **Model A (Baseline)**: `bert-base-chinese`  
- **Model B (Fine-tuned)**: `bert-base-chinese` fine-tuned on a subset of `ikala/tmmluplus`  
- **Judge Model**: `Qwen/Qwen2.5-3B-Instruct`

### Core Frameworks

- **PyTorch** – core deep learning framework  
- **Hugging Face Transformers** – model loading and inference  
- **DeepSpeed** – training acceleration and memory optimization  

---

## 📊 Evaluation Criteria

The Judge model evaluates responses across three dimensions:

1. **Correctness** – factual accuracy  
2. **Relevance** – alignment with the question  
3. **Completeness** – coverage of all key aspects  

Each model receives a **score from 1 to 10**, followed by a final comparative judgment.

---

## 📚 References

- **LLM-as-a-Judge Paper**  
  Judging LLM-as-a-judge with MT-Bench and Chatbot Arena  
  https://arxiv.org/abs/2306.05685

- **Qwen Model (Tongyi Qianwen)**  
  https://huggingface.co/Qwen
