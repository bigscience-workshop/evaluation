# BigScience Evaluation
Code and data for the [BigScience Evaluation WG](https://bigscience.huggingface.co/en/#!pages/working-groups.md).

## Quickstart

To benchmark a baseline GPT-2 model with WMT and TyDiQA datasets on GPU, run

```shell
python3 -m evaluation.eval \
    --model_name_or_path gpt2 \
    --eval_tasks wmt tydiqa_secondary \
    --device cuda \
    --output_dir outputs
```

## Setup

1. Create virtual environment (one-time).

   ```shell
   python3 -m venv venv # create a virtual environment called 'venv'
   ```
2. Activate the virtual environment.

   ```shell
   source venv/bin/activate
   ```

3. Install package requirements.

   ```shell
   python3 -m pip install -r requirements.txt
   ```
## Tasks

This project plans to support all datasets listed under `docs/datasets.md`.  The sections below detail task-independent inner-workings of this repository.

### AutoTask

Every task/dataset lives as a submodule within `evaluation.tasks`. The core of these submodules inherit from `evaluation.tasks.auto_task.AutoTask`, which is a base class that houses all abstract functions, as well has holds `model`, `tokenizer`, and `task_config` as its attributes. 

`AutoTask` makes it incredibly easy to load any dataset for a benchmark. The basic signature is

```python
task = AutoTask.from_task_name(
    "task_name", model, tokenizer, device, english_only
)
```

Alternatively, if the model has to be recreated for each task, a task object can be created from string specifications.

```python
task = AutoTask.from_spec(
    "task_name", 
    "model_name_or_path", 
    "tokenizer_name",
    device,
    english_only,
)
```

### Evaluation

Every `AutoTask` subclass has a `.evaluate()` function wherein all evaluation logic resides, i.e. loading the dataset (and the dataloader, if necessary), and computing reporting metrics. At the end of the evaluation, metrics are saved as a class attribute in `task.metrics`. For more details on the full pipeline, refer to the main evaluation script, [`evaluation/eval.py`](evaluation/eval.py). 

## Contributing

Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md).  
