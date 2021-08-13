from datasets import load_dataset
from torch.utils.data import Dataset


class WMTEnglishDataset(Dataset):
    def __init__(self, tokenizer, stride=512, max_len=1024, pair="kk-en"):
        super().__init__()
        assert (
            "en" in pair
        ), f"Expected `pair` to contain English, but got {pair} instead"
        wmt = load_dataset("wmt19", pair, split="validation")["translation"]
        text_list = [item["en"] for item in wmt]
        text = " ".join(text_list)
        input_ids = tokenizer(text, return_tensors="pt", verbose=False).input_ids.squeeze()
        self.input_ids = input_ids.unfold(size=max_len, step=stride, dimension=-1)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]
