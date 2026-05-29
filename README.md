<p align="center">
  <img src="./assets/Logo.png" alt="drawing" width="500"/>
</p>

**MedAssistant-8B** is a LoRA fine-tuned LLM designed for advanced medical reasoning. The model is able to assist with medical diagnosis by providing detailed explanations in Chain of Thought (CoT). 

- 🧠 Base Model: <a href="https://github.com/marketplace/models/azureml-meta/Meta-Llama-3-1-8B-Instruct">Llama-3.1-8B-Instruct</a>
- 🗂️ Dataset: <a href="https://github.com/BioMistral/BioMistral">MedBooks-CoT-18</a> & <a href="HPAI-BSC/Medprompt-MedQA-CoT">MedQA-CoT</a>
- 🛠️ LoRA Parameters: r=128, α=64

## Model

- Model Access: 🤗 <a href="https://github.com/marketplace/models/azureml-meta/Meta-Llama-3-1-8B-Instruct">MedAssistant-8B</a>
- Deploy: below provided a example code for direct inference with MedAssistant-8B.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('UCSC-VLAA/MedReason-8B',torch_dtype="auto",device_map="auto", use_safetensors= True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('UCSC-VLAA/MedReason-8B', trust_remote_code=True, padding_side='left')

input_text = "How to stop a cough?"
messages = [{"role": "user", "content": input_text}]

inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True), return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Piepline

Supervised-Finetuning (SFT) using LoRA improves the LLM’s medical reasoning capability. 
- Llama-3.1-8B-Instruct was fine-tuned on 4-GPU:

```bash
torchrun --nproc_per_node=4 train.py \
    --model_name_or_path 'meta-llama/Llama-3.1-8B-Instruct' \
    --model_max_length 2048 \
    --data_dir /path/to/your/data \
    --output_dir /path/to/output/dir \
    --resume_from_checkpoint True \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --is_lora True \
    --lora_rank 128 \
    --lora_alpha 64 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --optim "adamw_torch_fused" \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 4 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 1000 \
    --eval_strategy "steps" \
    --eval_steps 250 \
    --save_total_limit 3 \
    --report_to "none"
```

## Evaluation
- **Performance on Medical benchmarks**

|Benchmarks| Base Model | Fully Fine-tuned Model | LoRA Fine-tuned Model|
|----------|----------|----------|----------|
|   |<a href="https://github.com/marketplace/models/azureml-meta/Meta-Llama-3-1-8B-Instruct">Llama-3.1-8B-Instruct</a>| <a href="https://arxiv.org/abs/2504.00993">MedReason-8B</a>| MedAssist-8B   |
| **MedQA**   | 58.7%   | 71.8%   | 69.3%   |
| **MedBullet-5** (test accuracy) | 40.9%   | 55.5%   | 54.2%   |
| **MedBullet-5** (Rouge-L)  | 0.224  | N/A  | 0.344   |

- **Qualitative Results:**

  Case Study on Medbullets Benchmark. **MedAssistant-8B** generates accurate reasoning with reliable knowledge.

<img src="./assets/Case Study.png" alt="case_v6" style="zoom: 100%;" />




