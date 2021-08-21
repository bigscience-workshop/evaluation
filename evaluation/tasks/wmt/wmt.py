# Module for any additional processing required for the WMT dataset
# HuggingFace dataset link: https://huggingface.co/datasets/wmt19
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask


class WMTEnglishDataset(Dataset):
    def __init__(self, tokenizer, stride=512, max_len=1024, pair="kk-en"):
        super().__init__()
        assert "en" in pair, f"Expected `pair` to contain English, but got {pair} instead"
        wmt = load_dataset("wmt19", pair, split="validation")["translation"]
        text_list = [item["en"] for item in wmt]
        text = " ".join(text_list)
        input_ids = tokenizer(text, return_tensors="pt", verbose=False).input_ids.squeeze()
        self.input_ids = input_ids.unfold(size=max_len, step=stride, dimension=-1)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]


class WMTTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "wmt"

    def evaluate(self) -> None:
        stride = self.task_config["stride"]
        dataset = WMTEnglishDataset(
            self.tokenizer, stride=stride, max_len=self.model.config.n_positions, pair=self.task_config["pair"]
        )
        # TODO: resolve conflict with tokenizer to support num_workers
        loader = DataLoader(
            dataset,
            batch_size=self.task_config["batch_size"],
            shuffle=False,
            drop_last=True,
        )
        log_likelihoods = []
        for input_ids in tqdm(loader, desc=f"Evaluating {self.get_display_name()}"):
            input_ids = input_ids.to(self.device)
            target_ids = input_ids.clone()
            # Exclude context tokens from loss computation
            target_ids[:, :-stride] = -100
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                log_likelihood = outputs[0]
            log_likelihoods.append(log_likelihood)
        perplexity = torch.exp(torch.stack(log_likelihoods).sum() / len(loader))
        self.metrics["perplexity"] = perplexity.item()
