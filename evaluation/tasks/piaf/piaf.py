# HuggingFace dataset link: https://huggingface.co/datasets/piaf
import re
import string
from collections import Counter

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
    {{title}}{{"\n"}}{{context}}{{"\n"}}
    {%- endif -%}
    Given the {{_position}} {{_blank}}, answer the question: {{question}}
    {%- if  _position == "following" -%}
    {{"\n"}}{{title}}{{"\n"}}{{context}}
    {%- endif -%}
    {{"\n"}}Answer: 
    """  # noqa W291
)


class PIAFDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        assert tokenizer.pad_token == tokenizer.eos_token

        self.items = []

        piaf = load_dataset("piaf", split="train")
        for sample in piaf:
            prompt = TEMPLATE.render(
                id=sample["id"],
                title=sample["title"],
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
                    "lang": "french",
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                    "target_answer": sample["answers"]["text"],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# Evaluation of F1 and EM from the official SQuAD evaluate-v1.1.py script
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class PIAFTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "piaf"

    def evaluate(self) -> None:
        dataset = PIAFDataset(self.tokenizer)

        f1 = exact_match = substring_matches = 0
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
                [target_answer.lower() in predicted_answer.lower() for target_answer in target_answers]
            )
            substring_matches += substring_match

            exact_match += metric_max_over_ground_truths(exact_match_score, predicted_answer, target_answers)
            f1 += metric_max_over_ground_truths(f1_score, predicted_answer, target_answers)

        self.metrics = {
            "substring_matches": substring_matches / len(dataset) * 100,
            "exact_match": exact_match / len(dataset) * 100,
            "f1": f1 / len(dataset) * 100,
        }
