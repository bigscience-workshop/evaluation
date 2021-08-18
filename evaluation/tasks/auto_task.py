from abc import ABC, abstractmethod
from typing import Dict
import os

from transformers import AutoTokenizer

from evaluation.utils.io import save_json, load_json
from evaluation.models import load_model


class AutoTask(ABC):
    def __init__(
        self, 
        device, 
        english_only: bool, 
        model=None, 
        tokenizer=None,
        model_name_or_path="",  
        tokenizer_name="",
    ):
        assert model or model_name_or_path, "Expected either `model` or `model_name_or_path`"
        assert (
            tokenizer or tokenizer_name or model_name_or_path
        ), "Expected either `tokenizer` or `model_name_or_path` or `tokenizer_name`"
        if model is None:
            model = load_model(model_name_or_path).to(device)
        self.model = model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name_or_path)
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = {}
        self.task_config = self.load_task_args(english_only)

    @classmethod
    def from_task_name(
        cls, 
        task_name: str, 
        device, 
        english_only: bool, 
        model=None, 
        tokenizer=None,
        model_name_or_path="",  
        tokenizer_name="",
    ):
        all_tasks = cls.__subclasses__()
        for task in all_tasks:
            if task.get_display_name() == task_name:
                return task(
                    device=device, 
                    english_only=english_only,
                    model=model, 
                    tokenizer=tokenizer,
                    model_name_or_path=model_name_or_path,  
                    tokenizer_name=tokenizer_name,
                )
        
        raise ValueError(f'Invalid task: {task_name}')

    def load_task_args(self, english_only) -> Dict:
        task_root = os.path.join("evaluation", "tasks", self.get_display_name())        
        config_filename =  "english.json" if english_only else "multiligual.json"
        return load_json(os.path.join(task_root, config_filename))
    
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
