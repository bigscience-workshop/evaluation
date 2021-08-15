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
from evaluation import logger


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
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Directory for saving evaluation outputs."}
    )
    tag: Optional[str] = field(
        default=None,
        metadata={"help": "Identifier for the evaluation run."}
    )
    random_seed: int = field(
        default=24,
        metadata={"help": "Customized random seed"}
    )
    eval_tasks: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of tasks to run the evaluation on, e.g. tydiqa_secondary"}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device on which to run evaluation"}
    )


def main(args):
    if not args.eval_tasks:
        raise ValueError('Must provide at least one eval task!')
    
    logger.info("Beginning evaluation")

    # set random seed
    set_seed(args.random_seed)

    # initialize device
    device = torch.device(args.device)

    # Load model & tokenizer
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.eos_token)
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Exporting results
    tag = args.tag or datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    for eval_task in args.eval_tasks:
        logger.info(f"Benchmarking {eval_task}...")
        task = AutoTask.from_task_name(eval_task, tokenizer=tokenizer, model=model, device=device)
        task.evaluate()
        task.save_metrics(output_dir, logger)


if __name__ == "__main__":
    parser = HfArgumentParser(EvaluationArguments)
    args, = parser.parse_args_into_dataclasses()
    main(args)
