## Implementation

Students should implement the functions whose docstring begins with
`Implement:` in the template files under `code/template/`.

## clip

### Solution code

`clip/clip_solution.py`

### Data files

`clip/clip_gendata.py` generates:

- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`

Release `train/val` to students. Hide `test`.

### Autograder

`clip_solution.py --mode train` saves `outputs/clip_model.pt`.

Students run only `--mode train` locally and report the printed validation
numbers. Gradescope loads `outputs/clip_model.pt` and evaluates it on hidden
test data.

## vlm

### Solution code

`vlm/vlm_solution.py`

### Data files

`vlm/vlm_gendata.py` generates:

- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`
- `data/images/train/*.png`
- `data/images/val/*.png`
- `data/images/test/*.png`

Release `train/val` and the corresponding images to students. Hide `test`.

### Autograder

`vlm_solution.py --mode train` saves `outputs/vlm_model.pt`.

Students run only `--mode train` locally and report the printed zero-shot and
per-epoch validation numbers. Gradescope loads `outputs/vlm_model.pt` and
evaluates it on hidden test images and questions.

The released VLM uses `HuggingFaceTB/SmolVLM-500M-Instruct` with
`AutoProcessor`, `AutoModelForVision2Seq`, and LoRA fine-tuning.
