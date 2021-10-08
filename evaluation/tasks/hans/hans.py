from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
from jinja2 import Template
from evaluation.tasks.auto_task import AutoTask

TEMPLATE = Template(
    """
    {%- set _blank=["passage", "text", "text snippet", "context"]|random -%}
    {%- set _position = ["above", "following"] |random -%}
    {%- if  _position == "above" -%}
    {{"\n"}}{{context}}{{"\n"}}
    {%- endif -%}
    Given the {{_position}} {{_blank}}, answer the question: {{question}}
    {%- if  _position == "following" -%}
    {{"\n"}}{{"\n"}}{{context}}
    {%- endif -%}
    {{"\n"}}Answer:
    """ 
)


class HANSDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        hans = load_dataset("hans", split="validation")
        self.items = []
        self.heuristics = set()
        self.subcases = set()
        self.templates = set()

        for sample in hans:
            self.heuristics.add(sample["heuristic"])
            self.subcases.add(sample["subcase"])
            self.templates.add(sample["template"])
            prompt = TEMPLATE.render(
                    context = sample["premise"],
                    question = sample["hypothesis"] + " Is this True or False?"
                )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            )
            self.items.append(
                {
                    "prompt": prompt,
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                    "label": sample["label"],
                    "heuristic": sample["heuristic"],
                    "subcase": sample["subcase"],
                    "template": sample["template"]
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class HANSTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "hans"

    def evaluate(self) -> None:
        """
        All task-specific evaluation logic lives here.
        Model and tokenizer are available as self.model and self.tokenizer, respectively.
        For task-specific configurations, populate english.json or multilingual.json.
        Configs are read at initialization and available in dict form as self.task_config.
        For further details, refer to the AutoTask parent class in auto_task.py.
        """
        # Reference for metrics: https://github.com/tommccoy1/hans/blob/master/evaluate_heur_output.py
        dataset = HANSDataset(self.tokenizer)
        accuracy = 0.0
        heuristic_ent_correct_count_dict = {}
        subcase_correct_count_dict = {}
        template_correct_count_dict = {}
        heuristic_ent_incorrect_count_dict = {}
        subcase_incorrect_count_dict = {}
        template_incorrect_count_dict = {}
        heuristic_nonent_correct_count_dict = {}
        heuristic_nonent_incorrect_count_dict = {}

        for heuristic in dataset.heuristics:
            heuristic_ent_correct_count_dict[heuristic] = 0.0
            heuristic_ent_incorrect_count_dict[heuristic] = 0.0
            heuristic_nonent_correct_count_dict[heuristic] = 0.0
            heuristic_nonent_incorrect_count_dict[heuristic] = 0.0
        for subcase in dataset.subcases:
            subcase_correct_count_dict[subcase] = 0.0
            subcase_incorrect_count_dict[subcase] = 0.0
        for template in dataset.templates:
            template_correct_count_dict[template] = 0.0
            template_incorrect_count_dict[template] = 0.0

        for item in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            output = self.model.generate(
                input_ids=item["input_ids"].to(self.device),
                attention_mask=item["attention_mask"].to(self.device),
                max_length=min(item["input_len"] * 2, self.model.config.n_positions),
            )
            prompt_len = len(item["prompt"])
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            predicted_answer = decoded_output[prompt_len:].lower()
            correct_label = ["true", "false"][item["label"]]
            if correct_label in predicted_answer:
                accuracy += 1.0
                if correct_label == "true":
                    heuristic_ent_correct_count_dict[item["heuristic"]] += 1
                else:
                    heuristic_nonent_correct_count_dict[item["heuristic"]] += 1
                subcase_correct_count_dict[item["subcase"]] += 1
                template_correct_count_dict[item["template"]] += 1
            else:
                if correct_label == "true":
                    heuristic_ent_incorrect_count_dict[item["heuristic"]] += 1
                else:
                    heuristic_nonent_incorrect_count_dict[item["heuristic"]] += 1
                subcase_incorrect_count_dict[item["subcase"]] += 1
                template_incorrect_count_dict[item["template"]] += 1

        self.metrics["hans_overall_accuracy"] = accuracy / len(dataset) * 100
        for heuristic in dataset.heuristics:
            total = heuristic_ent_correct_count_dict[heuristic] + heuristic_ent_incorrect_count_dict[heuristic]
            self.metrics[f"hans_{heuristic}_entailed_accuracy"] = heuristic_ent_correct_count_dict[heuristic] / total * 100
            total = heuristic_nonent_correct_count_dict[heuristic] + heuristic_nonent_incorrect_count_dict[heuristic]
            self.metrics["hans_{}_nonentailed_accuracy".format(heuristic)] = heuristic_nonent_correct_count_dict[heuristic] / total * 100
        for subcase in dataset.subcases:
            total = subcase_correct_count_dict[subcase] + subcase_incorrect_count_dict[subcase]
            self.metrics["hans_{}_accuracy".format(subcase)] = subcase_correct_count_dict[subcase] / total * 100
        for template in dataset.templates:
            total = template_correct_count_dict[template] + template_incorrect_count_dict[template]
            self.metrics["template_{}_accuracy".format(template)] = template_correct_count_dict[template] / total * 100
