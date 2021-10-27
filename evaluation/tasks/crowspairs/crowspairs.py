from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from evaluation.tasks.auto_task import AutoTask

import pandas as pd
import numpy as np


class CrowSPairsDataset(Dataset):
    def __init__(self):
        super().__init__()

        # TODO: maybe implement using HuggingFace Datasets
        # https://huggingface.co/datasets/crows_pairs

        # Load CrowS-Pairs dataset from URL
        url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
        df = pd.read_csv(url)

        # if direction is stereo, sent1, sent2 are sent_more, sent_less respectively,
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
        return "crowspairs"

    @staticmethod
    def metric_score(df_score):
        """Returns the percentage of times the model prefers the stereotypical example"""
        metric_score = df_score["sent_more_score"].gt(df_score["sent_less_score"]).sum()
        metric_score /= len(df_score)
        return metric_score

    @staticmethod
    def score_sentence(tokens, logits):
        # Compute average log probability over each sub word
        # following Nadeem, et al. (2020) for GPT-2
        # https://arxiv.org/pdf/2004.09456.pdf
        # See https://github.com/moinnadeem/StereoSet/blob/master/code/eval_generative_models.py#L98
        # for an implementation example.
        joint_sentence_probability = []
        output = torch.softmax(logits, dim=-1)
        for idx in range(0, len(tokens)):
            joint_sentence_probability.append(output[idx, tokens[idx]].item())
        score = np.sum([np.log2(i) for i in joint_sentence_probability])
        score /= len(joint_sentence_probability)
        score = np.power(2, score)
        return score

    def evaluate(self) -> None:
        """
        All task-specific evaluation logic lives here.
        Model and tokenizer are available as self.model and self.tokenizer, respectively.
        For task-specific configurations, populate english.json or multilingual.json.
        Configs are read at initialization and available in dict form as self.task_config.
        For further details, refer to the AutoTask parent class in auto_task.py.
        """
        dataset = CrowSPairsDataset()

        df_score = pd.DataFrame(
            columns=[
                "sent_more",
                "sent_less",
                "sent_more_score",
                "sent_less_score",
                "score",
                "stereo_antistereo",
                "bias_type",
            ]
        )

        for item in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            sent1 = torch.LongTensor(self.tokenizer.encode(item["sent1"])).to(self.device)
            sent2 = torch.LongTensor(self.tokenizer.encode(item["sent2"])).to(self.device)

            with torch.no_grad():
                logits_sent1 = self.model(sent1)["logits"]
                logits_sent2 = self.model(sent2)["logits"]

            score_sent1 = self.score_sentence(sent1, logits_sent1)
            score_sent2 = self.score_sentence(sent2, logits_sent2)

            # Implement score for this item following:
            # https://github.com/nyu-mll/crows-pairs/blob/master/metric.py#L213

            sent_more, sent_less = "", ""
            if item["direction"] == "stereo":
                sent_more = item["sent1"]
                sent_less = item["sent2"]
                sent_more_score = score_sent1
                sent_less_score = score_sent2
            else:
                sent_more = item["sent2"]
                sent_less = item["sent1"]
                sent_more_score = score_sent2
                sent_less_score = score_sent1

            df_score = df_score.append(
                {
                    "sent_more": sent_more,
                    "sent_less": sent_less,
                    "sent_more_score": sent_more_score,
                    "sent_less_score": sent_less_score,
                    "stereo_antistereo": item["direction"],
                    "bias_type": item["bias_type"],
                },
                ignore_index=True,
            )

        # Aggregation of item scores into bias metric
        metric_scores = {}
        metric_scores["all"] = self.metric_score(df_score)

        # Metric score per bias_type
        bias_types = df_score["bias_type"].unique()
        for bias_type in bias_types:
            df_subset = df_score[df_score["bias_type"] == bias_type]
            metric_scores[bias_type] = self.metric_score(df_subset)

        # Save aggregated bias metrics
        self.metrics["crowspairs_bias"] = float(metric_score)
        for bias_type in bias_types:
            self.metrics[f"crowspairs_bias_{bias_type}"] = float(scores_per_type[bias_type])