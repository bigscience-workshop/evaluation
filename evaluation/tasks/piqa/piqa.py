# Module for any additional processing required for the TyDi QA dataset
# HuggingFace dataset link: https://huggingface.co/datasets/piqa
from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask

TEMPLATE = Template(
    """
Given a goal and 2 solutions, choose the most appropriate solution.
Goal: {{goal}}
{{'Solution 1'}}: {{sol1}}
{{'Solution 2'}}: {{sol2}}
Answer:
    """
)


class PIQADataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        piqa = load_dataset("piqa", split="validation")
        self.items = []

        for sample in piqa:
            prompt = TEMPLATE.render(
                goal=sample["goal"],
                sol1=sample["sol1"],
                sol2=sample["sol2"],
            )

            # Tokenize and construct this sample
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            )
            self.items.append(
                {
                    "prompt": prompt,
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                    "label": [sample["sol1"], sample["sol2"]][sample["label"]],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class PIQATask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "piqa"

    def evaluate(self) -> None:
        dataset = PIQADataset(self.tokenizer)

        substring_matches = 0
        for sample in tqdm(dataset):
            output = self.model.generate(
                input_ids=sample["input_ids"].to(self.device),
                attention_mask=sample["attention_mask"].to(self.device),
                max_length=min(sample["input_len"] * 2, self.model.config.n_positions),
            )
            prompt_len = len(sample["prompt"])
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            predicted_answer = decoded_output[prompt_len:]

            label = sample["label"]
            substring_match = int(label.lower() in predicted_answer.lower())

            substring_matches += substring_match

        self.metrics = {
            "substring_match": substring_matches / len(dataset) * 100,
        }
