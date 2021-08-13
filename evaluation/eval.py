import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import os

import torch
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import evaluation.tasks  # needed for AutoTask.__subclass__() to work correctly
from evaluation.tasks.auto_task import AutoTask


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
    output_dir: Optional[str] = field(
        default="outputs",
        metadata={"help": "Directory for saving evaluation outputs."}
    )
    random_seed: Optional[int] = field(
        default=24,
        metadata={"help": "Customized random seed"}
    )
    eval_tasks: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of tasks to run the evaluation on, e.g. tydiqa_secondary"}
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

    # set random seed
    set_seed(eval_args.random_seed)

    if not eval_args.eval_tasks:
        raise ValueError('Must provide at least one eval task!')

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

    # Exporting results
    output_dir = None
    if eval_args.output_dir:
        output_dir = os.path.join(eval_args.output_dir, datetime.now().strftime("%y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)

    for eval_task in eval_args.eval_tasks:
        logger.info(f"Benchmarking {eval_task}...")
        task = AutoTask.from_task_name(eval_task, tokenizer=tokenizer, model=model)
        task.evaluate()

        if output_dir:
            task.save_metrics(output_dir, logger)


if __name__ == "__main__":
    main()
