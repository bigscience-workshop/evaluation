import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from evaluation.datasets.tydiqa import TyDiQADataset

logger = logging.getLogger(__name__)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvaluationArguments:
    """
        Arguments for any adjustable params in this evaluation script
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint that we want to evaluate, could be name or the path."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )


def main():
    # parse arguments
    parser = HfArgumentParser(EvaluationArguments)
    eval_args, = parser.parse_args_into_dataclasses()

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)

    logger.info("Beginning evaluation")

    # Load model & tokenizer
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name or eval_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(eval_args.model_name_or_path, pad_token_id=tokenizer.eos_token)
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.to(torch_device)

    # Load dataset
    logger.info("Loading TyDiQA...")
    target_langs = ["english"]
    data = load_dataset("tydiqa", "secondary_task", split="validation")
    dataset = TyDiQADataset(data, tokenizer, target_langs)

    correct_tydiqa_answer = 0
    for sample in tqdm(dataset):
        output = model.generate(
            input_ids=sample["input_ids"].to(torch_device),
            attention_mask=sample["attention_mask"].to(torch_device),
            max_length=sample["input_len"] + 15,
        )

        prompt_len = len(sample["prompt"])
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted_answer = decoded_output[prompt_len:]
        target_answer = sample["target_answer"]
        prediction_contains_target_answer = target_answer in predicted_answer.lower()
        correct_tydiqa_answer += prediction_contains_target_answer
    correct_tydiqa_percentage = correct_tydiqa_answer / len(dataset) * 100
    print(f"{correct_tydiqa_percentage}% of samples answered correctly")


if __name__ == "__main__":
    main()
