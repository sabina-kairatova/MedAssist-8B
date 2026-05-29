import torch
import transformers
from torch.utils.data import Dataset
from jinja2 import Template
from pathlib import Path
import json
from random import shuffle
from torch.utils.data import random_split

IGNORE_INDEX = -100

class MedicalDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer       
        self.max_seq_len = max_seq_len
        messages_list = []

        chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        self.template = Template(tokenizer.chat_template)

        files = list(Path(data_dir).glob('*'))
        for file in files:
            if file.name == "medbooks-18-cot.jsonl":
                with file.open() as f:
                    lines = f.readlines()
                    messages_list.extend([json.loads(l)["messages"] for l in lines])
            elif file.name == "medprompt_medqa_cot.json":
                with file.open() as f:
                    data = json.load(f)
                    messages_list.extend([
                        [{"role": "system", "content": "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\"."}, 
                         {"role": "user", "content": f"{v['question']} {v['options']}"},
                         {"role": "assistant", "content": f"{v['generations'][0]['response']} {v['options'][v['correct_answer']]}"}] 
                         for _, v in data.items()
                    ])

        shuffle(messages_list)  
        processed_data = [self.preprocess(messages) for messages in messages_list]
        self.input_ids = [item["input_ids"] for item in processed_data]
        self.labels = [item["labels"] for item in processed_data]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def preprocess(self, messages):
        input_str = self.template.render(
            messages=messages, 
            bos_token=self.tokenizer.bos_token, 
            add_generation_prompt=False
        )
        query_str = self.template.render(
            messages=messages[:-1], 
            bos_token=self.tokenizer.bos_token, 
            add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
        query_ids = self.tokenizer.encode(query_str, add_special_tokens=False)
        labels = [IGNORE_INDEX] * len(query_ids) +  input_ids[len(query_ids):]
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:]}

    def collate_fn(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        max_len = max([len(b) for b in input_ids])
        input_ids = [instance + (max_len - len(instance)) * [self.tokenizer.eos_token_id] for instance in input_ids]
        labels = [instance + (max_len - len(instance)) * [IGNORE_INDEX] for instance in labels]
        return dict(
            input_ids=torch.LongTensor(input_ids),
            labels=torch.LongTensor(labels)
        )

def create_data_module(data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_len: int):
    dataset = MedicalDataset(data_path, tokenizer, max_seq_len)
    train_dataset, val_dataset = random_split(dataset, [0.95, 0.05])
    return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=dataset.collate_fn)


