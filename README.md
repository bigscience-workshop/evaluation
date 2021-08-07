# BigScience Evaluation
Code and data for the [BigScience Evaluation WG](https://bigscience.huggingface.co/en/#!pages/working-groups.md).

## Basic requirements
- Python 3.8
- [virtualenv](https://virtualenv.pypa.io/en/latest/)

## Setup
Run at the project root
### Virtual Environment
1. Create virtual environment (one-time)
```shell
python3 -m venv venv # create a virtual environment called 'venv'
```
2. Activate the virtual environment
```shell
source venv/bin/activate
```

### Install Requirements
```shell
python3 -m pip install -r requirements.txt
```

## Run evaluation scripts
### Simple Benchmark
A [simple benchmark](https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/22) that includes 
[WMT](https://huggingface.co/datasets/wmt19) and [TyDi QA](https://huggingface.co/datasets/tydiqa)
E.g.
```shell
python3 -m evaluation.scripts.simple_benchmark  --model_name_or_path=gpt2
```
