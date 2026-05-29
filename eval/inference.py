import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import json
import argparse
import torch
from tqdm import tqdm
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args() 


def load_data(file_path: str) -> List:
    data_list = []
    if os.path.basename(file_path) == "test.json":
        with open(file_path, "r") as fp:
            data = json.load(fp)["data"]
        for example in data:
            question = example["question"].strip()
            answer = example["answer"]
            prompt_dict = {
                    "prompt": f"""The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\"\n{question}\n""",
                    "gt_answer": answer
            }
            data_list.append(prompt_dict)
    elif os.path.basename(file_path) == "op5_test.parquet":
        df = pd.read_parquet(file_path, engine='auto')
        for _, row in df.iterrows():
            question = row["question"].strip()
            options = ", ".join([f"{key}: {value}" for key, value in row["options"].items()])
            gt_answer = row["answer"]
            gt_reasoning = row["explanation"]
            prompt_dict = {
                    "prompt": f"""The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\"\n{question} {options}\n""",
                    "gt_answer": gt_answer,
                    "gt_reasoning": gt_reasoning
            }
            data_list.append(prompt_dict)
    return data_list


def inference_on_one(input_str: List[str], model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
    inputs = tokenizer(input_str, return_tensors='pt', padding=True).to(device)
    output_tokens = model.generate(**inputs, max_new_tokens=500, top_k=10, pad_token_id=tokenizer.eos_token_id)
    output_str = tokenizer.batch_decode(output_tokens)
    return output_str[0]


def validate(model_path: str, data_path: str, output_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        model_max_length=2048, 
        use_fast=False, 
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    data_list = load_data(data_path)
    final_results = []
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        output_str = inference_on_one([data_entry["prompt"]], model, tokenizer)
        data_entry["output_str"] = output_str
        final_results.append(data_entry)
        with open(output_path, "a") as fp:
            fp.write('\n[SPLIT]\n')
            fp.write(data_entry["output_str"])
            fp.write("\nGround truth answer: " + data_entry["gt_answer"])
            if "gt_reasoning" in data_entry:
                fp.write("\nGround truth reasoning: " + data_entry["gt_reasoning"])


if __name__ == "__main__":
    args = parse_args()
    validate(model_path=args.model_path, data_path=args.data_path, output_path=args.output_path)