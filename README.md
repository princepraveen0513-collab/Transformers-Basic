# Parameter‑Efficient Fine‑Tuning (QLoRA) for Sentiment Classification with Transformers

**Notebook:** `Transformers_Intro.ipynb`

This project fine‑tunes a **causal language model** (e.g., EleutherAI/pythia-1b-deduped) on **Amazon Polarity** sentiment data using **QLoRA** (LoRA + 4‑bit quantization). The twist: instead of a standard classifier head, we **turn a generator into a classifier** via **instruction‑style prompts** (Instruction/Input/Response). This showcases how large LMs can be adapted for **supervised classification** with **tiny trainable parameter counts** and modest hardware.

---

## 🔎 What’s interesting here?
- **Instruction‑to‑classification**: sentiment labels are produced by prompting a **generative** model with a template (Instruction → Input → Response).  
- **QLoRA**: we keep the base model frozen, add **LoRA adapters** and train in **4‑bit** (bitsandbytes) — updating **~0.1–0.9%** of weights while preserving base knowledge.  
- **Small batches, real results**: batch size ≈ **4**, gradient accumulation **4**, sequence length **512**; learning rate tried: 1e-4, 2e-4, 3e-4; epochs: 1, 2.
- **Multiple runs**: several tuning runs with different LoRA ranks / dropout to compare quality & stability.

---

## 🧰 Environment & Setup
**Dependencies:** `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `torch`, `pandas`, `numpy`

Install (CUDA machine recommended):
```bash
pip install -U transformers datasets peft bitsandbytes accelerate torch pandas numpy
```

> If `bitsandbytes` isn’t available on your platform, the code falls back to standard precision (you may see a log like “BitsAndBytes not available. Falling back to standard precision.”).

---

## 🧠 Data & Prompting
- **Dataset:** `amazon_polarity` (Hugging Face Datasets) — detected.  
- Data is **formatted to an instruction prompt**:
  - `### Instruction:` “Classify the sentiment of the following product review.”  
  - `### Input:` the raw review text  
  - `### Response:` the **label** (e.g., `positive` / `negative`)

Tokenization uses a max sequence length of **512**, with truncation/padding as configured in the notebook.

---

## 🏗️ Model & PEFT (QLoRA)
- **Backbone:** EleutherAI/pythia-1b-deduped (loaded with `AutoModelForCausalLM`).  
- **Quantization:** 4‑bit (bnb) via `prepare_model_for_kbit_training`.  
- **LoRA config:** `r=16`, `lora_alpha=32`, `lora_dropout=0.05` (varied across runs).  
- **Trainable params:** the notebook logs examples like:
  ```
Sample tokenized input_ids: tensor([ 4118, 41959,    27,   187,  4947,  1419,   253, 21942,   273,   253])
trainable params: 2,097,152 || all params: 1,013,878,784 || trainable%: 0.2068
Step 10 | Loss: 14.6737
Step 20 | Loss: 14.4514
Step 30 | Loss: 14.2148
Step 40 | Loss: 13.6795
  ```

---

## ⚙️ Training
- **Batch size:** 4  ·  **Grad accum:** 4  ·  **Epochs:** 1, 2  
- **LR schedule:** constant LRs tried 1e-4, 2e-4, 3e-4  ·  **Weight decay:** 0.01
- **Logging:** every ~10 steps; example training loss curve is printed (steady decrease).  
- **Checkpoints:** saved under directories like `./tuning_run*` and `./fine_tuned_model/`.

---

## 📈 Quick Results (from notebook prints)
The notebook reports quick “sample” metrics for fast sanity checks (e.g., first 50 samples):
```
Sample tokenized input_ids: tensor([ 4118, 41959,    27,   187,  4947,  1419,   253, 21942,   273,   253])
trainable params: 2,097,152 || all params: 1,013,878,784 || trainable%: 0.2068
Step 10 | Loss: 14.6737
Step 20 | Loss: 14.4514
Step 30 | Loss: 14.2148
Step 40 | Loss: 13.6795
Step 50 | Loss: 12.5279
Step 60 | Loss: 10.8097
Step 70 | Loss: 9.4970
Step 80 | Loss: 8.5032
Step 90 | Loss: 7.6926
Step 100 | Loss: 7.0243
```
Interpret these as **rough health checks**, not final test accuracy. For a proper evaluation, run across the full validation/test split and compute **accuracy, precision, recall, F1**.

---

## 🧪 Inference
Load the base model and attach the trained LoRA weights, then generate a label via the same instruction template:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "EleutherAI/pythia-1b-deduped"
adapters_path = "./fine_tuned_model"  # or one of the tuning_run*/ checkpoints

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapters_path)

def classify(review_text):
    prompt = (
        "### Instruction:\nClassify the sentiment of the following product review.\n\n"
        f"### Input:\n{review_text}\n\n"
        "### Response:"
    )
    inputs = tokenizer(prompt.format(review_text=review_text), return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=3)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return "positive" if "positive" in text.lower() else ("negative" if "negative" in text.lower() else text.strip())

print(classify("This product exceeded my expectations!"))
```

---

## 📁 Repository Structure
```text
├── Transformers_Intro.ipynb
├── fine_tuned_model/                 # saved adapters/checkpoint(s)
├── tuning_run1/ ... tuning_run4*/    # comparison runs
└── README.md
```

---

## 🧭 Next Steps
- **Evaluate properly**: compute **accuracy, precision, recall, F1** on the full **test** split.  
- **LoRA search**: try `r` ∈ {8, 16, 32}, different `lora_dropout` and LRs; track **trainable %** vs quality.  
- **Prompt engineering**: refine the template and constrain the output space (e.g., `positive|negative`).  
- **Baselines**: compare against `AutoModelForSequenceClassification` (e.g., `distilbert-base-uncased`) on the same data.  
- **Export**: optionally `merge_and_unload()` adapters or create a lightweight **inference script**.

---

## 🔗 References
- Model: EleutherAI/pythia-1b-deduped  
- Dataset: **amazon_polarity** (Hugging Face Datasets)
