## Implementation

The students are required to implement all functions whose docstring
starts with `Implement`.

For release templates, remove the body of those functions and keep the
function signatures and docstrings.

These assignments will require both openAI API keys in .env file, and
GPUs. 

The students need to submit program outputs and report numbers by
themselves.

## SFT

This problem has two parts, under sft/

### Utility eval-data generator:
`sft/gen_eval_data.py` (not given to students)

The generated evaluation set under data/eval_dataset.json

### Part 1: Instruction Data generation

Solution code:

`sft/sft_p1_solution.py`

The students should generate data/sft_dataset.json which will be used
for instruction tuning of gpt model in part 2.

### Part 2: Supervised fine-tuning + judge evaluation

Solution code:

`sft/sft_p2_solution.py`

The students take ./data/sft_dataset.json from part 1, and finetune a
GPT2 model. After finetune, it will be evaluated using gpt5-nano.

We want win rate of sft over base to be at least 0.75 to get full
score. 

## RLHF

This problem has two parts, under rlhf/

We want both reward model test accuracies after training to be larger
than 0.58 to get full score.

### Part 1: Direct Reward model 

Solution code:

`hw4/code/rlhf/rlhf_p1_solution.py`

### Part 2: DPO implied Reward model

Solution code:

`hw4/code/rlhf/rlhf_p2_solution.py`

