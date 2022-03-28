import time
from itertools import zip_longest

import sacrebleu
import torch
from datasets import load_dataset
from jinja2 import Template
from sacrebleu.metrics import TER as _TER
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask


TEMPLATE = Template(
    """
    Translate graph to text:
    {{graph}}
    Verbalization:
    """
)


class WebNLGDataset(Dataset):
    def __init__(self, tokenizer, data_dir, dataset_split="test"):
        super().__init__()
        dataset = load_dataset("GEM/web_nlg", "en", split=dataset_split, data_dir=data_dir)
        self.items = []
        self.references = [sample["references"] for sample in dataset]
        for sample in dataset:
            prompt = TEMPLATE.render(graph=" ".join(sample["input"]))
            prompt = prompt.strip()

            inputs = tokenizer(prompt, padding=True, return_tensors="pt", truncation=True)

            self.items.append(
                {
                    "prompt": prompt,
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class WebNLGDatasetEval(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "webnlg"

    def evaluate_dataset(self, dataset_split):
        dataset = WebNLGDataset(self.tokenizer, self.data_dir, dataset_split=dataset_split)
        predictions = []
        self.model.eval()

        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=sample["input_ids"].to(self.device),
                    attention_mask=sample["attention_mask"].to(self.device),
                    max_length=self.task_config["max_generation_length"],
                    num_beams=self.task_config["num_beams"],
                    length_penalty=self.task_config["length_penalty"],
                )

            prompt_len = len(sample["prompt"])
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            predicted_answer = decoded_output[prompt_len:].strip()
            predictions.append(predicted_answer)

        ref_streams = list(zip_longest(*dataset.references))
        bleu = sacrebleu.corpus_bleu(predictions, ref_streams, lowercase=True)
        ter = self.ter_metric.corpus_score(predictions, ref_streams)
        self.metrics.update(
            {f"bleu_{dataset_split}": round(bleu.score, 5), f"ter_{dataset_split}": round(ter.score, 5)}
        )

    def evaluate(self) -> None:
        self.ter_metric = _TER(normalized=True, case_sensitive=False)
        self.time_start = time.time()
        self.evaluate_dataset("test")
        self.evaluate_dataset("challenge_test_scramble")
        self.evaluate_dataset("challenge_test_numbers")
        print("Total Run time", time.time() - self.time_start)
