# BigScience Evaluation
Code and data for the [BigScience Evaluation WG](https://bigscience.huggingface.co/en/#!pages/working-groups.md).

## Datasets included in the Full Benchmark
In July 2021, a vote was held to determine a short list for evaluating the final BigScience model. For more details,
see [slide deck](https://docs.google.com/presentation/d/1mvCcdYzA5jZgsDzwwGpOXrZvN4ygi-_M0Ubtuj-R_r0)

Below is the full list of datasets:
### MT
- [WMT](http://www.statmt.org/wmt20/metrics-task.html)
- [DiaBLa](https://github.com/rbawden/DiaBLa-dataset)

### NLU
- [SuperGLUE](https://super.gluebenchmark.com/)
- [TyDiQA](https://ai.google.com/research/tydiqa) (multilingual)
- [PIAF](https://github.com/etalab/piaf) (multilingual)
- [XQuAD](https://huggingface.co/datasets/xquad) (multilingual)

### NLG
- [Flores 101](https://github.com/facebookresearch/flores) (multilingual)
- [GEM](https://gem-benchmark.com/)
- [CRD3](https://huggingface.co/datasets/crd3)

### NER
- [MasakhaNER](https://github.com/masakhane-io/masakhane-ner) (multilingual)
- [WikiANN](https://github.com/afshinrahimi/mmner) (multilingual)

### Linguistic structure
- [BLiMP](https://github.com/alexwarstadt/blimp)
- [QA-SRL](https://qasrl.org/)  
- [UD](https://universaldependencies.org/) (multilingual)
- [LinCE](https://ritual.uh.edu/lince/) (multilingual)
- [LAMA](https://github.com/facebookresearch/LAMA)
- [Edge Probing](https://openreview.net/forum?id=SJzSgnRcKX)

### Few-shot
- [QASPER](https://allenai.org/data/qasper)
- [BioASQ](http://bioasq.org/)
- [TyDiQA](https://ai.google.com/research/tydiqa) (multilingual)
- [HuffPo](https://www.kaggle.com/rmisra/news-category-dataset)
- [MNLI](https://cims.nyu.edu/~sbowman/multinli/)
- [ANLI](https://github.com/facebookresearch/anli)
- [HANS](https://github.com/hansanon/hans)

### Social impact
- [WinoMT](https://github.com/gabrielStanovsky/mt_gender)
- [Jigsaw](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) 
- [CrowS-pairs](https://github.com/nyu-mll/crows-pairs/)
- TBD for Minimal Pair Tests

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
