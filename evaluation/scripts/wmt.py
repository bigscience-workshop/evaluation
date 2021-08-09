import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_wmt(tokenizer):
    wmt19 = load_dataset("wmt19", "kk-en")
    validation_set = wmt19["validation"]["translation"]
    text_list = [item["en"] for item in validation_set]
    text = " ".join(text_list)
    return tokenizer(text, return_tensors="pt")


def main(args):
    stride = args.stride
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    input_ids = load_wmt(tokenizer).input_ids.to(device)
    max_length = model.config.n_positions
    log_likelihoods = []
    total_length = input_ids.size(1)
    for i in tqdm(range(0, total_length, stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, total_length)
        trg_len = end_loc - i
        input_ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len
        log_likelihoods.append(log_likelihood)
    perplexity = torch.exp(torch.stack(lls).sum() / end_loc)
    print(perplexity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()
    main(args)
