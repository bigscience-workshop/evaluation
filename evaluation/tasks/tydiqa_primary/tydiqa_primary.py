# Module for any additional processing required for the TyDi QA dataset
# HuggingFace dataset link: https://huggingface.co/datasets/tydiqa

from jinja2 import Template
from torch.utils.data import Dataset

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
                prompt = prompt.strip()  # Remove trailing white space and newline

                # Tokenize and construct this sample
                inputs = tokenizer(
                    prompt,
                    padding=True,
                    return_tensors='pt',
                )
                self.items.append(
                    {
                        "prompt": prompt,
                        "lang": lang,
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "input_len": inputs["attention_mask"].shape[1],
                        "target_answer": [ans.lower() for ans in sample["answers"]['text']],
                    }
                )
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
