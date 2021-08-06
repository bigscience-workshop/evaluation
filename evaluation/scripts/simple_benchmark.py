import logging
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationArguments:
    """
        Arguments for any adjustable params in this evaluation script
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The model checkpoint that we want to evaluate, could be name or the path."}
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
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
    #tokenizer.pad_token = tokenizer.bos_token

    # Load dataset
    #target_langs = ["english"]
    #data = load_dataset("tydiqa", "secondary_task", split="validation")
    #dataset = TyDiQADataset(data, tm , tokenizer, target_langs)

if __name__ == "__main__":
    main()
