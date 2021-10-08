from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask

import pandas as pd


class CrowSPairsDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Load CrowS-Pairs dataset from URL
        url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
        df = pd.read_csv(url)
        # sent1, sent2 are sent_more, sent_less resp. if direction is stereo
        # otherwise the other way around
        df["direction"] = df["stereo_antistereo"]
        df["sent1"] = df["sent_less"]
        df["sent2"] = df["sent_more"]
        df.loc[df["direction"] == "stereo", "sent1"] = df["sent_more"]
        df.loc[df["direction"] == "stereo", "sent2"] = df["sent_less"]
        # Change dataframe to list of dictionaries
        self.items = df[["sent1", "sent2", "direction", "bias_type"]].to_dict("records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CrowSPairsTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "CrowS-Pairs"

    def evaluate(self) -> None:
        """
        All task-specific evaluation logic lives here.
        Model and tokenizer are available as self.model and self.tokenizer, respectively.
        For task-specific configurations, populate english.json or multilingual.json.
        Configs are read at initialization and available in dict form as self.task_config.
        For further details, refer to the AutoTask parent class in auto_task.py.
        """
        dataset = TemplateDataset()
        # NOTE: use torch.utils.data.DataLoader as needed
        for item in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            item = item.to(self.device)
            # TODO: write evaluation logic
        # TODO: replace some_metric with a metric name and save its value
        self.metrics["some_metric"] = 0
