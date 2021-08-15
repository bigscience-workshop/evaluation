# Module for any additional processing required for the TyDi QA dataset
# HuggingFace dataset link: https://huggingface.co/datasets/tydiqa
from typing import Dict

from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask

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
    def __init__(self, tokenizer, target_langs):
        super().__init__()
        tydiqa = load_dataset("tydiqa", "secondary_task", split="validation")
        self.items = []

        for sample in tydiqa:
            lang = sample["id"].split("-")[0]
            if lang in target_langs:
                # Filter out samples in languages that are not used during training
                prompt = TEMPLATE.render(
                    id=sample["id"],
                    context=sample["context"],
                    question=sample["question"],
                )
                prompt = prompt.strip()  # Remove trailing white space and newline

                # Tokenize and construct this sample
                inputs = tokenizer(
                    prompt,
                    padding=True,
                    return_tensors="pt",
                )
                self.items.append(
                    {
                        "prompt": prompt,
                        "lang": lang,
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "input_len": inputs["attention_mask"].shape[1],
                        "target_answer": [
                            ans.lower() for ans in sample["answers"]["text"]
                        ],
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TydiqaSecondaryTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "tydiqa_secondary"

    def evaluate(self) -> None:
        dataset = TyDiQADataset(self.tokenizer, target_langs=["english"])

        substring_matches = 0
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            output = self.model.generate(
                input_ids=sample["input_ids"].to(self.device),
                attention_mask=sample["attention_mask"].to(self.device),
                max_length=min(sample["input_len"] * 2, self.model.config.n_positions),
            )

            prompt_len = len(sample["prompt"])
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            predicted_answer = decoded_output[prompt_len:]

            target_answers = sample["target_answer"]
            substring_match = any(
                [
                    target_answer in predicted_answer.lower()
                    for target_answer in target_answers
                ]
            )
            substring_matches += substring_match

        self.metrics = {"substring_matches": substring_matches / len(dataset) * 100}
