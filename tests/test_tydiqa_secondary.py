from datasets import load_dataset
from promptsource.templates import TemplateCollection
from promptsource.utils import removeHyphen
from transformers import AutoTokenizer

from evaluation.tasks.tydiqa_secondary.tydiqa_secondary import TyDiQADataset


def test_prompt():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = TyDiQADataset(tokenizer, ["english"])
    prompt = next(iter(dataset))["prompt"]
    assert (
        "Wound care encourages and speeds wound healing via cleaning and protection from reinjury or infection. "
        "Depending on each patient's needs, it can range from the simplest first aid to entire nursing specialties "
        "such as wound, ostomy, and continence nursing and burn center care.\n"
    ) in prompt
    assert prompt.endswith("Answer:")


def test_promptsource_template():
    ds_key, sub_key = "tydiqa", "secondary_task"
    tydiqa_sec_vld_ds = load_dataset(ds_key, sub_key, split="validation", streaming=True)
    tydiqa_sec_vld_ds_en = filter(lambda x: x["id"].split("-")[0] == "english", tydiqa_sec_vld_ds)
    template_collection = TemplateCollection()
    tydiqa_sec_tmpls = template_collection.get_dataset(ds_key, sub_key)
    tmpl = tydiqa_sec_tmpls["simple_question_reading_comp_2"]
    prompt, _ = tmpl.apply(removeHyphen(next(tydiqa_sec_vld_ds_en)))
    assert (
        "Wound care encourages and speeds wound healing via cleaning and protection from reinjury or infection. "
        "Depending on each patient's needs, it can range from the simplest first aid to entire nursing specialties "
        "such as wound, ostomy, and continence nursing and burn center care.\n"
    ) in prompt
