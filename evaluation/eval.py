import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

import evaluation.tasks  # noqa: F401; needed for AutoTask.__subclass__() to work correctly
from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger


@dataclass
class EvaluationArguments:
    """
    Arguments for any adjustable params in this evaluation script
    """

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint that we want to evaluate, could be name or the path."}
    )
    eval_tasks: List[str] = field(metadata={"help": "A list of tasks to run the evaluation on, e.g. tydiqa_secondary"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    tag: Optional[str] = field(default=None, metadata={"help": "Identifier for the evaluation run."})
    english_only: Optional[bool] = field(default=True, metadata={"help": "Whether to run evaluation in English only."})


def main():
    parser = HfArgumentParser((EvaluationArguments, TrainingArguments))
    eval_args, train_args = parser.parse_args_into_dataclasses()

    if not eval_args.eval_tasks:
        raise ValueError("Must provide at least one eval task!")

    # initialize device
    device = torch.device(train_args.device)

    logger = get_logger()
    logger.info(f"Beginning evaluation on device {train_args.device}")

    # Load model & tokenizer
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name or eval_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        eval_args.model_name_or_path,
        pad_token_id=tokenizer.eos_token,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Exporting results
    tag = eval_args.tag or datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join(train_args.output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    for eval_task in eval_args.eval_tasks:
        logger.info(f"Benchmarking {eval_task}...")
        task = AutoTask.from_task_name(
            eval_task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            english_only=eval_args.english_only,
        )
        set_seed(train_args.seed)
        task.evaluate()
        task.save_metrics(output_dir, logger)


if __name__ == "__main__":
    main()
