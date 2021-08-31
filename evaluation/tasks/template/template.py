from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask


class TemplateDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO: load and process dataset
        # can use load_dataset() in HF datasets
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TemplateTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        # TODO: replace some_task with proper display name
        return "some_task"

    def evaluate(self) -> None:
        """
        All task-specific evaluation logic lives here.
        Model and tokenizer are available as self.model and self.tokenizer, respectively.
        For task-specific configurations, populate english.json or multilingual.json.
        Configs are read at initialization and available in dict form as self.task_config.
        For further details, refer to the AutoTask parent class in auto_task.py.
        """
        dataset = TemplateDataset()
        # NOTE: use torch.utils.data.DataLoader as needed
        for item in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            item = item.to(self.device)
            # TODO: write evaluation logic
        # TODO: replace some_metric with a metric name and save its value
        self.metrics["some_metric"] = 0
