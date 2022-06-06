# Module for any additional processing required for the ANLI dataset
# HuggingFace dataset link: https://huggingface.co/datasets/anli
import torch
from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask


TEMPLATE = Template(
    """
    {{premise}}
      Question: {{hypothesis}} True, False, or Neither? ||| {{ answer_choices[label]
      }}
    """
)

prompt_dict = {0: "True", 1: "Neither", 2: "False"}

splits = ["dev_r1", "dev_r2", "dev_r3"]


class ANLIDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        anli = load_dataset("anli")
        self.items = []
        for split in splits:
            for sample in anli[split]:
                prompt = TEMPLATE.render(
                    premise=sample["premise"],
                    hypothesis=sample["hypothesis"],
                    answer_choices=prompt_dict,
                    label=sample["label"],
                )
                prompt = prompt.strip()  # a space at the front indicating that the target is a word
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
                self.items.append(
                    {
                        "prompt": prompt,
                        "input_ids": inputs["input_ids"][0][:-1],
                        "attention_mask": inputs["attention_mask"][0][:-1],
                        "input_len": inputs["attention_mask"].shape[1],
                        "target": sample["label"],
                        "split": split,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class ANLITask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "anli"

    def evaluate(self) -> None:
        dataset = ANLIDataset(self.tokenizer)

        matches = dict.fromkeys(splits, 0)  # counter for each split
        lens = dict.fromkeys(splits, 0)  # different length for each split
        target_options = [self.tokenizer.encode(word)[0] for word in prompt_dict.values()]
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            with torch.no_grad():
                logits = self.model(
                    input_ids=sample["input_ids"].to(self.device),
                    attention_mask=sample["attention_mask"].to(self.device),
                )["logits"]
            target_logits = logits[-1]
            prediction = target_logits[target_options].argmax().item()
            matches[sample["split"]] += prediction == sample["target"]
            lens[sample["split"]] += 1

        self.metrics = {f"accuracy_{split}": matches[split] / lens[split] for split in splits}
