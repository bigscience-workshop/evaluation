from transformers import AutoModelForCausalLM


def load_model(model_name_or_path):
    return AutoModelForCausalLM.from_pretrained(model_name_or_path)
