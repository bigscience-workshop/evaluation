from torch.utils.data import Dataset
from datasets import load_dataset


class LAMBADADataset(Dataset):
    def __init__(self):
        super().__init__()
        lambada = load_dataset("lambada", split="validation")
        items = []
        for item in lambada:
            text = item["text"]
            text_list = text.split()
            target = text_list[-1]
            input_ = " ".join(text_list[:-1])
            items.append([input_, target])
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]