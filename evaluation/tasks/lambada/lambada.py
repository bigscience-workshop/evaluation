# Module for any additional processing required for the LAMBADA dataset
# HuggingFace dataset link: https://huggingface.co/datasets/lambada
import numpy as np
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask


class LAMBADADataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        lambada = load_dataset("lambada", split="validation")
        self.items = []

        for sample in lambada:
            # Split to context and target
            text = sample["text"]
            context = text.rsplit(" ", 1)[0]
            target = " " + text.rsplit(" ", 1)[1]  # a space at the front indicating that the target is a word

            # Tokenize and construct this sample
            context_tokenized = tokenizer.encode(context)
            target_tokenized = tokenizer.encode(target)
            input_ids = (context_tokenized + target_tokenized)[:-1]

            self.items.append(
                {
                    "input_ids": torch.LongTensor(input_ids),
                    "label": torch.LongTensor(target_tokenized),
                    "label_len": len(target_tokenized),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class LAMBADATask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "lambada"

    def evaluate(self) -> None:
        dataset = LAMBADADataset(self.tokenizer)

        loss_fn = CrossEntropyLoss(reduction="sum")
        num_predictions = 0
        all_matches = 0
        losses = []
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            input_ids = sample["input_ids"].to(self.device)
            label = sample["label"].to(self.device)

            with torch.no_grad():
                all_logits = self.model(input_ids)["logits"]  # logits of the whole sequence (i.e. context + target)
                target_logits = all_logits[-sample["label_len"] :]  # logits of the target (i.e. last word)
            predictions = target_logits.argmax(dim=-1)

            num_predictions += sample["label_len"]
            loss = loss_fn(target_logits, label).detach().cpu().item()
            all_match = (predictions == label).all()

            losses.append(loss)
            all_matches += int(all_match)

        perplexity = np.exp(sum(losses) / num_predictions)
        self.metrics = {
            "perplexity": perplexity,
            "accuracy": all_matches / len(dataset) * 100,
        }
