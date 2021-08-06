# Module for any additional processing required for the TyDi QA dataset
# HuggingFace dataset link: https://huggingface.co/datasets/tydiqa

import torch
from torch.utils.data import Dataset
from jinja2 import Template

TEMPLATE = Template(
    """
    {%- set _blank=["passage", "text", "text snippet", "context"]|random -%}
    {%- set _position = ["above", "following"] |random -%}
    {%- if  _position == "above" -%}
    {{context}}{{"\n"}}
    {%- endif -%}
    Given the {{_position}} {{_blank}}, answer the question: {{question}}
    {%- if  _position == "following" -%}
    {{"\n"}}{{context}}
    {%- endif -%}
    {{"\n"}}Answer: 
    """
)

class TyDiQADataset(Dataset):
    def __init__(self, data, tokenizer, target_langs):
        super(TyDiQADataset, self).__init__()
        self.items = []
        
        for sample_id, sample in enumerate(data):
            lang = sample["id"].split("-")[0]
            if lang in target_langs:
                # Filter out samples in languages that are not used during training
                prompt = TEMPLATE.render(
                    id       = sample["id"],
                    context  = sample["context"],
                    question = sample["question"],
                )
                prompt = tokenizer.bos_token + " " + prompt.strip()  # Remove trailing white space and newline

                # Tokenize and construct this sample
                inputs = tokenizer(prompt, max_length=tokenizer.model_max_length,  padding="max_length") 
                label_ids = [tokenizer(answer)["input_ids"] for answer in sample["answers"]["text"]]
                self.items.append(
                    {
                        "prompt": prompt,
                        "lang": lang,
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "label_ids": label_ids,
                        "input_len": sum(inputs["attention_mask"]),
                        "answers": sample["answers"],
                    }
                )
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        item = self.items[index]
        return {
            "prompt": item["prompt"],
            "lang": item["lang"],
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "label_ids": item["label_ids"],
        }

