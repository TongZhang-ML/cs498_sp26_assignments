## Implementation

The students are required to implement all functions whose docstring
starts with `Implement`.

For release templates, remove the body of those functions and keep the
function signatures and docstrings.

## SFT

This subproblem has two parts.

### Utility eval-data generator:
`sft/gen_eval_data.py` (not given to students)

### Part 1: Instruction Data generation

Solution code:

`sft/sft_p1_solution.py`


### Part 2: Supervised fine-tuning + judge evaluation

Solution code:

`sft/sft_p2_solution.py`

We want win rate of sft over base to be over 0.7 

## RLHF

This subproblem has two parts.

Notes:

- Expected reward model test accuracy after training should roughly achieve ` 0.6`.

### Part 1: Reward model training

Solution code:

`hw4/code/rlhf/rlhf_p1_solution.py`

### Part 2: DPO training

Solution code:

`hw4/code/rlhf/rlhf_p2_solution.py`

