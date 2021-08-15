import os
from abc import ABC, abstractmethod

from evaluation.utils.io import save_json


class AutoTask(ABC):
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.metrics = {}

    @classmethod
    def from_task_name(cls, task_name: str, tokenizer, model, device):
        all_tasks = cls.__subclasses__()
        for task in all_tasks:
            if task.get_display_name() == task_name:
                return task(tokenizer=tokenizer, model=model, device=device)

        raise ValueError(f"Invalid task: {task_name}")

    @staticmethod
    @abstractmethod
    def get_display_name() -> str:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass

    def save_metrics(self, output_dir, logger=None) -> str:
        output_filename = os.path.join(output_dir, f"{self.get_display_name()}.json")
        save_json(self.metrics, output_filename)

        if logger:
            logger.info(
                f"{self.get_display_name()}: result exported to {output_filename}"
            )
        return output_filename
