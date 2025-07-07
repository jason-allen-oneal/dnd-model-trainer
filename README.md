# 🐉 Llama 3 D&D Fine-Tuning

This repository contains a complete Colab pipeline to fine-tune a **Llama 3 8B** language model on a custom **Dungeons & Dragons 5e dataset**, using **[Unsloth](https://github.com/unslothai/unsloth)** for fast training with **LoRA**, **bitsandbytes**, and **GGUF export**.

---

## 📚 Dataset

- **Source**: [`jason-oneal/dnd-5e-dataset`](https://huggingface.co/datasets/jason-oneal/dnd-5e-dataset)  
- Format: Instruction / Input / Output (Alpaca-style)

---

## 🧩 Model

- **Base**: `unsloth/llama-3-8b-bnb-4bit`  
  A pre-quantized 4-bit version for low VRAM usage.
- Fine-tuning with LoRA adapters.

---

## ⚙️ Installation

Inside Colab:

```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
pip install --no-deps unsloth
```

---

## 🔑 Authentication

Set your **Hugging Face tokens** as Colab secrets:

```python
from google.colab import userdata

HF_READ = userdata.get('HF_READ')   # read token
HF_TOKEN = userdata.get('HF_TOKEN') # write token

from huggingface_hub import login
login(token=HF_TOKEN)
```

---

## 🧵 Fine-Tuning

### 1️⃣ Load the base model with Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    token = HF_READ,
)
```

### 2️⃣ Add a LoRA adapter

```python
peft_model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### 3️⃣ Load and format the dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "jason-oneal/dnd-5e-dataset",
    split="train",
    token=HF_READ,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    return {
        "text": [
            alpaca_prompt.format(i, inp, out) + EOS_TOKEN
            for i, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
        ]
    }

dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)
```

### 4️⃣ Train with TRL

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=300,
        learning_rate=5e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer.train()
```

---

## 🚀 Model Export

After training, push all formats to the Hub:

```python
REPO = "jason-oneal/Llama-3-8B-DnD"

peft_model.push_to_hub_merged(f"{REPO}-merged-16bit", tokenizer, save_method="merged_16bit", token=HF_TOKEN)
peft_model.push_to_hub_merged(f"{REPO}-merged-4bit", tokenizer, save_method="merged_4bit_forced", token=HF_TOKEN)
peft_model.push_to_hub_merged(f"{REPO}-lora", tokenizer, save_method="lora", token=HF_TOKEN)

peft_model.push_to_hub_gguf(f"{REPO}-8bit-Q8_0-gguf", tokenizer, token=HF_TOKEN)
peft_model.push_to_hub_gguf(f"{REPO}-16bit-gguf", tokenizer, quantization_method="f16", token=HF_TOKEN)
peft_model.push_to_hub_gguf(f"{REPO}-q4_k_m-gguf", tokenizer, quantization_method="q4_k_m", token=HF_TOKEN)

# Final merge and push
model = peft_model.merge_and_unload()
model.eval()

model.push_to_hub(REPO, token=HF_TOKEN)
tokenizer.push_to_hub(REPO, token=HF_TOKEN)
```

---

## ⚡ Tips

- If you run out of VRAM, switch to `load_in_4bit=False` for full precision.
- Adjust `max_steps` and `batch_size` for your budget.
- Always inspect your dataset with `.head()` or `.take(5)`!

---

## 📢 License

Check your base model and dataset license before redistributing.
