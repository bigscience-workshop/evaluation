import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from evaluation.datasets.tydiqa import TyDiQADataset

logger = logging.getLogger(__name__)


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

    logger.info('Beginning evaluation')

    # Load model & tokenizer
    logger.info('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name or eval_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(eval_args.model_name_or_path, pad_token_id=tokenizer.eos_token)

    model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    logger.info('Loading TyDiQA...')
    target_langs = ["english"]
    data = load_dataset("tydiqa", "secondary_task", split="validation")
    dataset = TyDiQADataset(data, tokenizer, target_langs)

    for sample in dataset:
        output = model.generate(
            input_ids=sample['input_ids'],
            attention_mask=sample['attention_mask'],
            max_length=tokenizer.model_max_length + 1,
            num_beams=4,
            early_stopping=True,
        )
        prompt_len = len(sample['prompt'])
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded_output[prompt_len:]
        print(answer)

if __name__ == "__main__":
    main()
