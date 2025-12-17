## 使用 Qwen 模型評估回答質量指南
本專案使用 Qwen-3B 模型作為評估器（Judge），來比較微調後的 BERT 模型和原始模型的回答質量。這是一種 LLM-as-a-Judge 的評估方法。

---
### 🎯 核心原理
LLM-as-a-Judge 的核心思想是利用一個強大、中立的第三方大型語言模型（在此專案中為 Qwen），來自動化評估其他模型（微調前後的 BERT）的輸出品質。傳統上，這類評估需要大量的人力來進行主觀評分，而 LLM-as-a-Judge 提供了一種更快速、可擴展且成本較低的替代方案。

評估者（Judge）模型會接收一個包含以下內容的提示（Prompt）：
1.  原始問題。
2.  一個參考的「標準答案」。
3.  模型 A（原始 BERT）的回答。
4.  模型 B（微調後 BERT）的回答。

然後，評估者模型會根據預設的標準（如正確性、相關性、完整性）對兩個模型的回答進行評分，並判斷哪一個更好。

---
### 🔍 工作原理
整個評估流程如下：
1.  **生成基準測試集**：從 `ikala/tmmluplus` 資料集中隨機抽取問題，形成一個標準化的問答基準測試集 `qa_benchmark`。
2.  **模型回答**：
    *   **模型 A (Baseline)**：使用原始的 `bert-base-chinese` 模型，對測試集中的每個問題生成回答。
    *   **模型 B (Fine-tuned)**：使用在 `ikala/tmmluplus` 資料集上微調過的 BERT 模型，對相同的問題生成回答。
3.  **構建評估提示 (Prompt)**：為每個問題，動態生成一個結構化的 Prompt，將問題、標準答案、以及兩個模型的回答整合在一起。
4.  **啟動評估**：將此 Prompt 發送給作為「評估者」的 Qwen 模型。
5.  **獲取評估結果**：Qwen 模型會輸出對兩個模型回答的評分、質性評論，並宣告哪一個模型表現更優。

---
### 代碼結構說明
專案主要由 `fine_tuning_llm_round1.ipynb` 構成，其關鍵組件如下：

- **`generate_qa_benchmark()`**: 從 Hugging Face Hub 加載資料集，並隨機抽樣生成評估用的問答對。
- **`get_dataset()`**: 負責加載和預處理用於微調 BERT 模型的訓練資料。
- **訓練循環**: 使用 `DeepSpeed` 框架對 `bert-base-chinese` 模型進行高效的微調訓練。
- **`chat_with_tuning_llm()`**: 用於與**微調後**的模型互動，並獲取其回答。
- **`general_chat()`**: 用於與**原始**的 `bert-base-chinese` 模型互動，獲取其回答作為基準。
- **`EvalLLm` 類**:
    - `__init__`: 初始化評估者模型（Qwen），並加載兩個待評估模型的回答和標準答案。
    - `evaluation()`: 核心評估函數，負責構建評估 Prompt，並從 Qwen 模型獲取評估結果。
- **主評估循環**: 遍歷 `qa_benchmark` 中的每一個問題，調用上述函數，最終輸出每個問題的評估報告。

---
### prompt 設計
為了引導評估者模型做出公正且結構化的評判，我們設計了以下的 Prompt 模板：

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
- **`{question}`**: 原始問題。
- **`{gold_answer}`**: 標準參考答案。
- **`{baseline_output}`**: 原始模型的回答。
- **`{finetuned_output}`**: 微調模型的回答。
- **指令**: `Act as an impartial judge...` 這部分明確指示評估者模型扮演一個公正的角色，並根據具體標準進行評分。

---
### 配置選項
- **`model_name` (in `EvalLLm`)**: 評估者模型的名稱，預設為 `Qwen/Qwen2.5-3B-Instruct`。您可以將其更換為其他強大的模型，如 `gpt-4` 或 `claude-3-opus`。
- **`config_params` (in notebook)**: DeepSpeed 的訓練配置，包括 `train_batch_size`, `lr` (學習率) 等，可根據硬體資源進行調整。
- **`random_seed` (in `generate_qa_benchmark`)**: 用於確保每次生成的評估測試集都是相同的，以保證評估的可複現性。

---
### 技術細節
- **模型 A (基準模型)**: `bert-base-chinese`
- **模型 B (微調模型)**: `bert-base-chinese` 在 `ikala/tmmluplus` 的部分資料上進行微調後的版本。
- **評估者模型 (Judge Model)**: `Qwen/Qwen2.5-3B-Instruct`
- **主要框架**:
    - `PyTorch`: 核心深度學習框架。
    - `Hugging Face Transformers`: 用於加載和操作預訓練模型。
    - `DeepSpeed`: 用於加速和優化模型訓練過程。

---
### 評估標準
評估者模型被要求從以下三個維度進行評分：
1.  **正確性 (Correctness)**: 回答的內容是否事實準確？
2.  **相關性 (Relevance)**: 回答是否直接針對問題？有沒有偏題？
3.  **完整性 (Completeness)**: 回答是否全面，涵蓋了問題的所有重要方面？

最終，評估者會為每個模型提供 1-10 分的評分，並給出一個總結性的判斷。

---
### 參考資料
- **LLM-as-a-Judge 論文**: [Judging LLM-as-a-judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- **Qwen (通義千問) 模型**: [Hugging Face Model Page](https://huggingface.co/Qwen)
- **DeepSpeed 官方網站**: [DeepSpeed Website](https://www.deepspeed.ai/)

---
