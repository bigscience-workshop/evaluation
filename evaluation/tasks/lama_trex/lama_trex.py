from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask


class LAMA_Trex_Dataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        # load trex dataset
        lama = load_dataset("lama", "trex", split="train")

        self.items = []

        triples_added = set()
        for sample_id, sample in enumerate(lama):
            id = (sample["uuid"],)
            obj_label = (sample["obj_label"],)
            sub_label = sample["sub_label"]
            template = sample["template"]
            predicate_id = sample["predicate_id"]
            template = template.strip()  # Remove trailing white space and newline

            # adapt the [MASK ]template to work with a causal LM
            # we cut off the remaining part of the template. this may cause problems for some LAMA templates
            template = template.replace("[X]", sub_label)
            template = template.split("[Y]")[0]
            triple = (sub_label, predicate_id, obj_label)

            # Tokenize and construct this sample
            inputs = tokenizer(template, padding=True, return_tensors="pt",)
            if triple not in triples_added:
                triples_added.add(triple)
                self.items.append(
                    {
                        "template": template,
                        "lang": "eng",
                        "id": id,
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "input_len": inputs["attention_mask"].shape[1],
                        "target_answer": obj_label[0],
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class LAMA_Trex_Task(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "lama_trex"

    def evaluate(self) -> None:

        dataset = LAMA_Trex_Dataset(self.tokenizer)
        # NOTE: use torch.utils.data.DataLoader as needed

        # count the number of correct answers
        counter = 0
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            output = self.model.generate(
                input_ids=sample["input_ids"].to(self.device),
                attention_mask=sample["attention_mask"].to(self.device),
                max_length=min(sample["input_len"] * 2, self.model.config.n_positions),
            )
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = decoded_output.split(".")[0].replace(sample["template"], "")

            target_answer = sample["target_answer"]

            # this step is kind of different from the original LAMA evaluation, since it checks whether the correct answer is within a number of predicted words.
            if target_answer in prediction:
                counter += 1

        self.metrics["precision@1"] = counter / len(dataset)
