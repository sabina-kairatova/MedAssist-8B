<p align="center">
  <img src="./assets/Logo.png" alt="drawing" width="500"/>
</p>

**MedAssistant-8B** is a LoRA fine-tuned LLM designed for advanced medical reasoning. The model is able to assist with medical diagnosis by providing detailed explanations in Chain of Thought (CoT). 

- 🧠 Base Model: <a href="https://github.com/marketplace/models/azureml-meta/Meta-Llama-3-1-8B-Instruct">Llama-3.1-8B-Instruct</a>
- 🗂️ Dataset: <a href="https://github.com/BioMistral/BioMistral">MedBooks-CoT-18</a> & <a href="HPAI-BSC/Medprompt-MedQA-CoT">MedQA-CoT</a>
- 🛠️ LoRA Parameters: r=128, α=64
- ⚙️ Hardware: 4 x A40 GPUs

## Model

- Model Access: 🤗 <a href="https://huggingface.co/rare-engineer/MedAssistant-8B">MedAssistant-8B</a>
- Deploy: below provided a example code for direct inference with MedAssistant-8B.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('rare-engineer/MedAssistant-8B',torch_dtype="auto",device_map="auto", use_safetensors= True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('rare-engineer/MedAssistant-8B', trust_remote_code=True, padding_side='left')

user_instruction = "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer."
input_text = "A 17-year-old girl presents to the emergency department with a headache. The patient has had headaches in the past but this is the worst headache of her life. Her symptoms started yesterday and have been getting progressively worse. The patient states that the pain is mostly on the left side of her head. There has been a recent outbreak of measles at the patient’s school and the patient’s mother has been trying to give her daughter medicine to prevent her from getting sick. Her mother fears that her daughter may have caught measles. Her temperature is 98.6°F (37°C), blood pressure is 123/74 mmHg, pulse is 85/min, and respirations are 13/min. On exam, the patient is an obese girl who is clutching her head with the light in the room turned off. Her neurological exam is within normal limits. Fundoscopic exam reveals mild bilateral papilledema. An MRI of the head is obtained and reveals cerebral edema. A lumbar puncture reveals an increased opening pressure with a normal glucose level. Which of the following is the most likely diagnosis? A: Bacterial meningitis, B: Fat-soluble vitamin overuse, C: Migraine headache, D: Subarachnoid hemorrhage, E: Viral meningitis"

inputs = tokenizer([user_instruction + '\n'+ input_text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
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

