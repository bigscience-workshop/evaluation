from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask

from .task_names import blimp_task_names


class BLIMPDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.items = [
            load_dataset("blimp", task, split="train") for task in blimp_task_names[:2]
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class BLIMPTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "blimp"

    def evaluate(self) -> None:
        dataset = BLIMPDataset()
        num_correct = 0
        num_items = 0

        for task_dataset in dataset:
            for sample in tqdm(
                task_dataset,
                desc=f"Evaluating {self.get_display_name()} - {task_dataset.config_name}",
            ):
                tokenized_good = self.tokenizer(
                    sample["sentence_good"], return_tensors="pt"
                )["input_ids"]
                tokenized_bad = self.tokenizer(
                    sample["sentence_bad"], return_tensors="pt"
                )["input_ids"]

                logits_good = self.model(
                    input_ids=tokenized_good.to(self.device),
                ).logits
                logits_bad = self.model(
                    input_ids=tokenized_bad.to(self.device),
                ).logits

                # Compute sentence log probabilities from full LM probability distribution
                log_prob_good = logits_good[
                    0, range(tokenized_good.shape[1] - 1), tokenized_good[0, 1:]
                ].sum()
                log_prob_bad = logits_bad[
                    0, range(tokenized_bad.shape[1] - 1), tokenized_bad[0, 1:]
                ].sum()

                if log_prob_good > log_prob_bad:
                    num_correct += 1

                num_items += 1

        self.metrics["accuracy"] = num_correct / num_items
