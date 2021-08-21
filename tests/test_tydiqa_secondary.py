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
