import os

!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
!pip install --no-deps unsloth

from google.colab import userdata
HF_READ = userdata.get('HF_READ')
HF_TOKEN = userdata.get('HF_TOKEN')
REPO = "jason-oneal/Llama-3-8B-DnD"

from huggingface_hub import login
login(token=HF_TOKEN)

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF_READ,
)

# PEFT model (LoRA)
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
    use_rslora = False,
    loftq_config = None,
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

from datasets import load_dataset
dataset = load_dataset(
    "jason-oneal/dnd-5e-dataset",
    split="train",
    token=HF_READ,
)
dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
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

trainer_stats = trainer.train()

# Push all formats BEFORE merge
peft_model.push_to_hub_merged(f"{REPO}-merged-16bit", tokenizer, save_method="merged_16bit", token=HF_TOKEN)
peft_model.push_to_hub_merged(f"{REPO}-merged-4bit", tokenizer, save_method="merged_4bit_forced", token=HF_TOKEN)
peft_model.push_to_hub_merged(f"{REPO}-lora", tokenizer, save_method="lora", token=HF_TOKEN)

peft_model.push_to_hub_gguf(f"{REPO}-8bit-Q8_0-gguf", tokenizer, token=HF_TOKEN)
peft_model.push_to_hub_gguf(f"{REPO}-16bit-gguf", tokenizer, quantization_method="f16", token=HF_TOKEN)
peft_model.push_to_hub_gguf(f"{REPO}-q4_k_m-gguf", tokenizer, quantization_method="q4_k_m", token=HF_TOKEN)

# Now merge and push the plain merged model
model = peft_model.merge_and_unload()
model.eval()

model.push_to_hub(REPO, token=HF_TOKEN)
tokenizer.push_to_hub(REPO, token=HF_TOKEN)
