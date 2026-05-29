import torch
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import dataclass, field
from transformers import(
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    Trainer
)
from typing import List
from prepare_dataset import create_data_module

SEED = 42

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    model_max_length: int = field(default=None)
    is_lora: bool = field(default=False)
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

@dataclass
class DataArguments:
    data_dir: str = field(default=None)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        use_fast=False,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if model_args.is_lora:
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    data_module = create_data_module(data_args.data_dir, tokenizer, model_args.model_max_length)
    
    trainer = Trainer(
        model,
        args=training_args,
        **data_module
    )

    trainer.train()
    model.save_pretrained("./final_result")


if __name__ == "__main__":
    main()

