from abc import ABC, abstractmethod
import os

import torch

from evaluation.utils.io import save_json


class AutoTask(ABC):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = {}

    @classmethod
    def from_task_name(cls, task_name: str, tokenizer, model):
        all_tasks = cls.__subclasses__()
        matched_task = [task for task in all_tasks if task.get_display_name() == task_name]

        if not matched_task:
            raise ValueError(f'Invalid task: {task_name}')

        return matched_task[0](tokenizer=tokenizer, model=model)

    @staticmethod
    @abstractmethod
    def get_display_name() -> str:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass

    def save_metrics(self, output_dir, logger=None) -> str:
        # Exporting TyDiQA results
        output_filename = os.path.join(output_dir, f"{self.get_display_name()}.json")
        save_json(self.metrics, output_filename)

        if logger:
            logger.info(f"{self.get_display_name()}: result exported to {output_filename}")
        return output_filename
