import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..datasets.wmt import WMTEnglishDataset


def main(args):
    """adapted from https://huggingface.co/transformers/perplexity.html"""
    stride = args.stride
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    dataset = WMTEnglishDataset(
        tokenizer, stride=stride, max_len=model.config.n_positions, pair=args.pair,
    )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=True,
    )
    log_likelihoods = []
    for input_ids in tqdm(loader):
        input_ids = input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0]
        log_likelihoods.append(log_likelihood)
    perplexity = torch.exp(torch.stack(log_likelihoods).sum() / len(loader))
    print(f"perplexity: {perplexity.item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--pair", type=str, default="kk-en")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
