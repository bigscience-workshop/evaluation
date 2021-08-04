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


if __name__ == "__main__":
    main()
