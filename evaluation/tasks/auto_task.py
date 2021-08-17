from abc import ABC, abstractmethod
from typing import Dict
import os

from transformers import AutoTokenizer

from evaluation.utils.io import save_json, load_json
from evaluation.models import load_model


class AutoTask(ABC):
    def __init__(
        self, model_name_or_path, device, is_english_only, tokenizer_name,
    ):
        self.model = load_model(model_name_or_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name_or_path)
        self.device = device
        self.metrics = {}
        self.task_config = self.load_task_args(is_english_only)

    @classmethod
    def from_task_name(
        cls, task_name: str, model_name_or_path, device, is_english_only, tokenizer_name="",
    ):
        all_tasks = cls.__subclasses__()
        for task in all_tasks:
            if task.get_display_name() == task_name:
                return task(
                    model_name_or_path=model_name_or_path, 
                    device=device, 
                    tokenizer_name=tokenizer_name, 
                    is_english_only=is_english_only,
                )
        
        raise ValueError(f'Invalid task: {task_name}')

    def load_task_args(self, is_english_only) -> Dict:
        task_root = os.path.join("evaluation", "tasks", self.get_display_name())        
        if is_english_only:
            return load_json(os.path.join(task_root, "english.json"))
        return load_json(os.path.join(task_root, "multiligual.json"))
    
    @staticmethod
    @abstractmethod
    def get_display_name() -> str:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass

    def train(self) -> None:
        # TODO: convert to `abstractmethod` once simple_benchmark is ready
        raise NotImplementedError

    def save_metrics(self, output_dir, logger=None) -> str:
        output_filename = os.path.join(output_dir, f"{self.get_display_name()}.json")
        save_json(self.metrics, output_filename)

        if logger:
            logger.info(f"{self.get_display_name()}: result exported to {output_filename}")
        return output_filename
