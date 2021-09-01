# Contributing

Everyone is welcome to contribute, and we value contributions from the community. One of the best ways to contribute is by adding a data set to the evaluation benchmark!

## How to “claim” a task
1. Find a task on the [issues](https://github.com/bigscience-workshop/evaluation/issues) page. Self-assign or comment to indicate interest.
2. Coordinate when more than 1 contributor have indicated interests.
3. Open a new branch
4. Open a pull request "Add <task_name> dataset" when you are ready. Make sure to include
   1. which model(s) the task was evaluated on
   2. computation time benchmark on GPU (preferred) and/or CPU

## How to add a task via the task template
1. New tasks will be placed under `evaluation/tasks`
2. Make a copy of the directory `evaluation/tasks/template` and rename the directory to match your task, i.e. in the root directory, run
```shell script
cp -r evaluation/tasks/template evaluation/tasks/{{SOME_NEW_TASK}}
```
3. Your new task directory will include 4 files: 
   1. `__init__.py`
   2. `english.json`: json file for task-specific configurations of english-only data (e.g. batch_size)
   3. [For multilingual tasks only] `multilingual.json`: json file for task-specific configuration of multilingual data
   4. `task_name.py`: the main module

## What to implement in the task template
1. Wrap data as Pytorch Dataset/DataLoader
2. Rename TemplateTask (which inherits `AutoTask`) to match your task
3. Implement all abstract your task

References:
- [Template task](https://github.com/bigscience-workshop/evaluation/blob/main/evaluation/tasks/template/template.py)
- Fully implemented example for [TydiQA Secondary](https://github.com/bigscience-workshop/evaluation/blob/main/evaluation/tasks/tydiqa_secondary/tydiqa_secondary.py)

## Other notes on development
1. Feel free to use Hugging Face's GPT2LMHead as the base model
2. Make modifications and commit any changes. It's best to make your commit messages informative to help your reviewer. Below is a few list of meta-labels to get you started.
```
#    feat     (new feature)
#    fix      (bug fix)
#    refactor (refactoring production code)
#    style    (formatting, missing semi colons, etc; no code change)
#    docs     (changes to documentation)
#    test     (adding or refactoring tests; no production code change)
#    chore    (updating grunt tasks etc; no production code change)
#    build    (changes that affect the build system or external dependencies)
#    ci       (changes to our CI configuration files and scripts)
#    version  (version bump/new release; no production code change)
#    debug    (Changes in debugging code/frameworks; no production code change)
#    license  (Edits regarding licensing; no production code change)
#    hack     (Temporary fix to make things move forward; please avoid)
```
For example, one possible commit message would be `feature: implement lambada evaluation`.
3. Write prompts to reformat the dataset to LM task if necessary (e.g. QA tasks)
   1. Submit prompts to the [promptsource](https://github.com/bigscience-workshop/promptsource/blob/main/CONTRIBUTING.md) repo
   2. Prompts are in jinja2 format 
   3. Try to have at least 3 prompts
4. Run `make quality` at the roof of the repo to check for linting and code styling issues
5. Run `make style` at the root of the repo to auto-format the code

## After contributing to the repo
- Update the [Overleaf Tech Report](https://www.overleaf.com/8547355528ksstrmgjbfmj) with information on the task you added
- Add a new Github issue requesting your task be made [multilingual](https://github.com/bigscience-workshop/evaluation/labels/multilingual)
  - Label the issue with “multilingual”
  - Specify in the text of the issue which languages the task already supports
  - The multilinguality group is working on recruiting speakers of all the training languages to adapt English prompts to other languages
